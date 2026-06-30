import numpy as np
import pandas as pd
import math
import os
import shutil
import datetime
import pyabf
import copy
from Autonanopore import Autonanopore_train_control
from scipy.stats import gumbel_r
from scipy.stats import kstest
from numba import jit

#EWMAによるイベント検出
#EWMAutoクラスに呼び出される関数
@jit(nopython=True, cache=True)
def EWMA_search(data, baseline_std, exponential_weight=0.9999):
    weight = exponential_weight
    event_frag = False
    current_mean: float = data[0]
    std: float = baseline_std
    local_mean: np.ndarray = [current_mean for _ in range(len(data))]
    event_region = []
    mean_eventstart = data[0]
    event_start_index = 0
    event_end_index = 0

    for i in range(1, len(data)):
        if event_frag:
            if data[i] < mean_eventstart:
                local_mean[i] = current_mean
                continue
            else:
                event_end_index = i
                event_frag = False
                event_region.append([event_start_index, event_end_index])
        else:
            if data[i] > local_mean[i - 1] - 6 * std:
                current_mean = weight * current_mean + (1 - weight) * data[i]
            else:
                event_start_index = i
                event_frag = True
                mean_eventstart = current_mean
        local_mean[i] = current_mean
    
    #イベントの始点が閾値を超えた場所からになっているので、イベントの始点をdataがlocal_meanを上回る場所に変更
    boundary_search_failed = []

    for i in range(len(event_region)):
        for j in range(event_region[i][0]-1, 0, -1):
            if data[j] > local_mean[j]:
                event_region[i][0] = j
                break
        else:
            boundary_search_failed.append(i)
    
    for i in boundary_search_failed[::-1]:
        del event_region[i]

    #np.save(os.path.join(output_path, 'event_region.npy'), event_region)
    return event_region

#thresholdはgumbel分布にフィッティングした結果から求める
class EWMAuto:
    def __init__(self, path_control, data_scale_control=1, whether_blockage_control=True, sampling_rate=250000, section_length=30, detection_thr=7, max_threshold=200, log=True):#コントロールがなければNoneを代入
        #各種メンバ変数の初期化
        self.sampling_rate = sampling_rate
        self.section_length = section_length
        self.detection_thr = detection_thr
        self.max_threshold = max_threshold
        self.log = log

        #controlからの閾値の計算、コントロールがない場合は0を代入
        if path_control is not None:
            self.control_trainer = Autonanopore_train_control(path_control, data_scale_control, whether_blockage_control, sampling_rate, section_length)
            control_data = self.read_data(path_control, data_scale_control, whether_blockage_control)
            self.baseline_std = float(np.std(control_data[:1000]))
        else:
            self.control_trainer = (0, 0)
            self.baseline_std = None
        
        if self.log:
            if path_control == None:
                print("No control data")
            elif self.control_trainer[0] > 0.01:
                print("Fitting control data succeeded")
            else:
                print("Fitting control data failed")
        
    #実行
    def run(self, path_data, data_scale=1, whether_blockage_data=True, output_folder_name="EWMAuto_output", output_file_prefix="EWMAuto", EWMA_weight=0.9999, save_trace=True, max_events=None):
        if self.log:
            start_time = datetime.datetime.now()
            print("Start time: ", start_time)
        
        if self.log:
            print("Start reading data...")
        #測定データの読み込み
        self.data = self.read_data(path_data, data_scale, whether_blockage_data)#単位はpA
        if self.log:
            print("Data reading completed")

        #メンバ変数の初期化
        self.output_file_prefix = output_file_prefix
        self.members_initialize(path_data, output_folder_name)

        baseline_std = self.baseline_std if self.baseline_std is not None else float(np.std(self.data[:1000]))
        event_region = EWMA_search(data=self.data, baseline_std=baseline_std, exponential_weight=EWMA_weight)
        if self.log:
            print("EWMA search completed")
        if len(event_region) != 0:
            sections = self.event_covered_section(event_region)
        else:
            sections = [[i * self.section_length_points, (i+1) * self.section_length_points] for i in range(self.num_sections)]

        results = self.process_sections(sections)#Pi, Ki, Bi, Ai, section_length
        event_index, threshold = self.detect_events(results, sections)
        if self.log:
            print("Event detection completed")

        eventdata = self.calculate_eventdata(self.event_search(sections, event_index, threshold, results))
        if max_events is not None and len(eventdata) > max_events:
            if self.log:
                print(f"Event count ({len(eventdata)}) exceeds max_events={max_events}. Truncating.")
            eventdata = eventdata[:max_events]
        if self.log:
            print("Event characterization completed")

        self.save_result(eventdata, threshold)
        if save_trace:
            self.save_currents(eventdata)
            self.save_features_csv(eventdata)
            if self.log:
                print("Results saved")

        if self.log:
            print(f"{len(eventdata)} events detected")
            print(f"frequency = {len(eventdata) / (len(self.data) / self.sampling_rate)} /s")

            print("Total time: ", datetime.datetime.now() - start_time, "\n", "\n")

    def members_initialize(self, path_data, output_folder_name):
        self.output_path = os.path.join(os.path.dirname(path_data), output_folder_name)
        self.datamean = np.mean(self.data)
        self.section_length_points = round(self.sampling_rate * self.section_length / 1000)
        self.point_interval = round(1000000 / self.sampling_rate)  # us
        self.num_sections = len(self.data) // self.section_length_points
        self.final_event_contain_section = len(self.data) // self.section_length_points


        #binファイルとabfファイルの読み込みに対応
        #それ以外のファイル形式の場合はエラーを出力
        #本当はエラーが出た段階でプログラムを終了させたい
    def read_data(self, path_data, scale, mode):
        starttime = datetime.datetime.now()
        filetype = os.path.splitext(path_data)[1]
        if filetype == '.bin':
            with open(path_data, mode='rb') as f:
                data = f.read()
            if mode == True:
                return np.frombuffer(data, np.float64)*scale#単位はpA
            else:
                return -np.frombuffer(data, np.float64)*scale#単位はpA
        elif filetype == '.abf':
            abf = pyabf.ABF(path_data)
            abf.setSweep(0)
            if mode == True:
                return np.array(abf.sweepY, dtype=np.float64)*scale#単位はpA
            else:
                return -np.array(abf.sweepY, dtype=np.float64)*scale#単位はpA
        else:
            print("Error: Invalid filetype")
        print("Data reading completed")
        print("Time to read data: ", datetime.datetime.now() - starttime)


        #EWMAによるイベント検出
    
    #各セクションに含まれる、EWMAによって検出されたイベントが1つ以下になるように、元のセクションを調整している。なるべく元のセクションは保持するようになっている
    #元のセクションを1つずつ見ていったとき、イベントとの位置関係は6通りある。それぞれのパターンに対して、セクションの境界を調整する。
    #調整した後のセクションを新しいセクションとして返す
    #イベントを含まないセクションは[セクションの始点, セクションの終点]の形で返す。イベントを含むセクションは[セクションの始点, セクションの終点, イベントの始点, イベントの終点]の形で返す
    def event_covered_section(self, event_region):
        default_section = [[i * self.section_length_points, (i+1) * self.section_length_points - 1] for i in range(self.num_sections)]#そのまま30 msごとに分割したセクション
        indice_event_region = 0 #event_regionのインデックス
        new_section = [] #境界を調整したセクションを格納するリスト

        #次に位置関係を考えるイベントの始点と終点
        next_event_start = event_region[indice_event_region][0]
        next_event_end = event_region[indice_event_region][1]

        #次に位置関係を考えるセクションの始点
        next_section_start = default_section[0][0]

        #現在のイベントの始点と終点
        current_event_start = event_region[0][0]
        current_event_end = event_region[0][1]

        for section in default_section:
            #pattern 0 or 1
            if indice_event_region >= len(event_region) or next_event_start >= section[1]:
                new_section.append(section)

            #pattern 2 or 4
            elif next_event_start >= section[0]:
                current_event_start = copy.copy(next_event_start)

                #pattern 2
                if next_event_end >= section[1]:
                    next_section_start = section[0]
                    #merge frag

                #pattern 4
                else:
                    current_section_start = section[0]
                    current_section_end = section[1]
                    current_event_start = copy.copy(next_event_start)
                    current_event_end = copy.copy(next_event_end)
                    indice_event_region += 1
                    if indice_event_region >= len(event_region):
                        new_section.append([section[0], section[1], next_event_start, next_event_end])
                        continue
                    next_event_start = event_region[indice_event_region][0]
                    next_event_end = event_region[indice_event_region][1]

                    while section[1] > next_event_end:
                        current_section_end = (current_event_end + next_event_start) // 2
                        new_section.append([current_section_start, current_section_end, current_event_start, current_event_end])
                        current_section_start = current_section_end+1
                        current_event_start = copy.copy(next_event_start)
                        current_event_end = copy.copy(next_event_end)
                        indice_event_region += 1
                        if indice_event_region >= len(event_region):
                            break
                        next_event_start = event_region[indice_event_region][0]
                        next_event_end = event_region[indice_event_region][1]
                    
                    if next_event_start > section[1]:
                        new_section.append([current_section_start, section[1], current_event_start, current_event_end])
                    else:
                        current_section_end = (current_event_end + next_event_start) // 2
                        new_section.append([current_section_start, current_section_end, current_event_start, current_event_end])
                        next_section_start = current_section_end
                        #to pattern 3 or 5  

            #pattern 3 or 5
            else:
                #pattern 3
                if next_event_end >= section[1]:
                    continue

                #pattern 5
                else:
                    current_event_start = copy.copy(next_event_start)
                    current_event_end = copy.copy(next_event_end)
                    current_section_start = next_section_start
                    current_section_end = section[1]
                    indice_event_region += 1
                    if indice_event_region >= len(event_region):
                        new_section.append([next_section_start, section[1], next_event_start, next_event_end])
                        continue
                    next_event_start = event_region[indice_event_region][0]
                    next_event_end = event_region[indice_event_region][1]
                    while section[1] > next_event_end:
                        current_section_end = (current_event_end + next_event_start) // 2
                        new_section.append([current_section_start, current_section_end, current_event_start, current_event_end])
                        current_section_start = current_section_end+1
                        current_event_start = copy.copy(next_event_start)
                        current_event_end = copy.copy(next_event_end)
                        indice_event_region += 1
                        if indice_event_region >= len(event_region):
                            break
                        next_event_start = event_region[indice_event_region][0]
                        next_event_end = event_region[indice_event_region][1]
                    if next_event_start > section[1]:
                        new_section.append([current_section_start, section[1], current_event_start, current_event_end])
                    else:
                        current_section_end = (current_event_end + next_event_start) // 2
                        new_section.append([current_section_start, current_section_end, current_event_start, current_event_end])
                        next_section_start = current_section_end
        
        #最後のセクションがイベントを含む場合、そのセクションのインデックスを保存
        if len(event_region) > 0:
            for i in range(len(new_section)-1, 0, -1):
                if len(new_section[i]) == 4:
                    self.final_event_contain_section = i
                    break 
        else:
            self.final_event_contain_section = len(new_section) - 1

        return new_section
    
    #各セクションの処理を行う
    def process_sections(self, sections):
        results = []
        for i in range(len(sections)):
            results.append(self.calculate_sections(sections[i]))
        print("Section processing completed")
        return results
    
   
    #各セクションに対して、ピーク値、ピーク値のインデックス、ベースライン、ピーク値からベースラインを引いた値を計算する
    def calculate_sections(self, section):
        if len(section) == 2:
            section_start, section_end = section
        elif len(section) == 4:
            section_start, section_end, event_start, event_end = section
        section_data = self.data[section_start:section_end]
        Ki = np.argmin(section_data)
        Pi = section_data[Ki]

        if len(section) == 2:
            cutrange = round((self.section_length // 6) * self.sampling_rate / 1000)#デフォルトでは前後5ms
            if len(section_data) < cutrange * 2:
                Bi = np.mean(section_data)
            elif Ki <= cutrange:
                Bi = np.mean(section_data[Ki+cutrange:])
            elif Ki >= len(section_data) - cutrange:
                Bi = np.mean(section_data[:Ki-cutrange])
            else:
                Bi = np.mean(np.concatenate((section_data[:Ki-cutrange], section_data[Ki+cutrange:])))
        elif len(section) == 4:
            if event_start - section_start <= 0 and event_end - section_start >= len(section_data):#sectionのほぼ全てがイベントで占められている場合
                Bi = (section_data[0] + section_data[-1]) / 2
            else:
                Bi = np.mean(np.concatenate([section_data[:(event_start-section_start)], section_data[(event_end-section_start):]]))
        Ai = Bi - Pi
        Ki = Ki + section_start
        
        if section_end - section_start > self.section_length_points:
            return [Pi, Ki, Bi, Ai, section_end - section_start]

        return [Pi, Ki, Bi, Ai, False]
    
    #各セクションにイベントが存在するかを判定する
    # def detect_events(self, results, sections):
    #     amplitudes = [r[3] for r in results]
    #     recalc_flag = [r[4] for r in results]

    #     if self.control_trainer[0] > 0.01:
    #         threshold = gumbel_r.ppf(0.95**(1/len(results)), loc=self.control_trainer[1][0], scale=self.control_trainer[1][1])
    #     else:
    #         threshold, param = self.calculate_threshold_without_control(sections)
    #     thresholds = []

    #     event_index = []
    #     for i, j in zip(range(len(amplitudes)), recalc_flag):
    #         if j == False:
    #             if amplitudes[i] > threshold:
    #                 event_index.append(i)
    #                 thresholds.append(threshold)
    #         else:
    #             if self.control_trainer[0] > 0.01:
    #                 threshold_temp = self.recalculate_threshold(j, self.control_trainer[1])
    #             else:
    #                 threshold_temp = self.recalculate_threshold(j, param)
    #             if amplitudes[i] > threshold_temp:
    #                 event_index.append(i)
    #                 thresholds.append(copy.copy(threshold_temp))
    #     return event_index, thresholds
    def detect_events(self, results, sections):
        amplitudes = [r[3] for r in results]
        recalc_flag = [r[4] for r in results]

        if self.control_trainer[0] > 0.01:
            threshold = gumbel_r.ppf(
                0.50 ** (1 / len(results)),
                loc=self.control_trainer[1][0],
                scale=self.control_trainer[1][1]
            )
            param = self.control_trainer[1]
        else:
            threshold, param = self.calculate_threshold_without_control(sections)

        event_index = []
        thresholds = []

        for i, flag in enumerate(recalc_flag):
            # IQR法では再計算しない
            thr = threshold

            if amplitudes[i] > thr:
                event_index.append(i)
                thresholds.append(thr)

        return event_index, thresholds

    #長いセクションでの閾値の再計算
    def recalculate_threshold(self, section_length, fitting_param):
        beta_few = fitting_param[1]
        mu_few = fitting_param[0]
        beta_many = beta_few * np.sqrt(((section_length+1) * self.section_length_points * math.log(self.section_length_points)) / ((self.section_length_points+1) * section_length * math.log(section_length)))
        mu_part = lambda n: (np.sqrt(2*(n+1)*math.log(n) / n) - ((math.log(math.log(n)) + math.log(4 * np.pi))) / ((2*np.sqrt(2*n*math.log(n)/(n+1)))))
        mu_many = (mu_part(section_length) / mu_part(self.section_length_points)) * mu_few
        new_threshold = gumbel_r.ppf(0.50 ** (1 / 10000), loc=mu_many, scale=beta_many)
        return new_threshold
    
    # def calculate_threshold_without_control(self, sections):
    #     amplitudes = []
    #     for section in sections:
    #         if len(section) == 2:#EWMAでイベントが検出されず長さが30msのセクション
    #             section_start, section_end = section
    #             section_data = self.data[section_start:section_end]
    #             Ki = np.argmin(section_data)
    #             Pi = section_data[Ki]

    #             cutrange = round((self.section_length // 6) * self.sampling_rate / 1000)#デフォルトでは前後5ms
    #             if len(section_data) < cutrange * 2:
    #                 Bi = np.mean(section_data)
    #             elif Ki <= cutrange:
    #                 Bi = np.mean(section_data[Ki+cutrange:])
    #             elif Ki >= len(section_data) - cutrange:
    #                 Bi = np.mean(section_data[:Ki-cutrange])
    #             else:
    #                 Bi = np.mean(np.concatenate((section_data[:Ki-cutrange], section_data[Ki+cutrange:])))
    #             Ai = Bi - Pi
    #             if Ai < self.max_threshold:#あまりに大きいものはイベントであるとして除外
    #                 amplitudes.append(Ai)
    #     param = gumbel_r.fit(amplitudes)
    #     pv = kstest(amplitudes, 'gumbel_r', args=param)[1]
    #     print("p-value of fitting amplitudes of noevent region", pv)
    #     threshold = gumbel_r.ppf(0.95**(1/len(sections)), loc=param[0], scale=param[1])
    #     return threshold, param
    def calculate_threshold_without_control(self, sections):
        amplitudes = []
        
        for section in sections:
            # EWMAでイベントが検出されなかった通常セクションのみ使用
            if len(section) == 2:
                section_start, section_end = section
                section_data = self.data[section_start:section_end]

                Ki = np.argmin(section_data)
                Pi = section_data[Ki]

                cutrange = round(
                    (self.section_length // 6) * self.sampling_rate / 1000
                )

                if len(section_data) < cutrange * 2:
                    Bi = np.mean(section_data)
                elif Ki <= cutrange:
                    Bi = np.mean(section_data[Ki + cutrange:])
                elif Ki >= len(section_data) - cutrange:
                    Bi = np.mean(section_data[:Ki - cutrange])
                else:
                    Bi = np.mean(
                        np.concatenate((
                            section_data[:Ki - cutrange],
                            section_data[Ki + cutrange:]
                        ))
                    )
                

                Ai = Bi - Pi

                if Ai < self.max_threshold:
                    amplitudes.append(Ai)

        amplitudes = np.asarray(amplitudes)

        if len(amplitudes) < 10:
            raise ValueError("Not enough data for IQR threshold estimation")

        Q1 = np.percentile(amplitudes, 25)
        Q3 = np.percentile(amplitudes, 75)
        IQR = Q3 - Q1

        threshold = Q3 + 7* IQR  # 式(35)  変更

        if self.log:
            print(
                f"[IQR threshold] Q1={Q1:.3f}, Q3={Q3:.3f}, "
                f"IQR={IQR:.3f}, threshold={threshold:.3f}"
            )

        return threshold, None


    #イベントを含むセクションについてイベント境界を決定
    #同じセクションに他に閾値を超える点が存在するかを確認
    #存在した場合そのイベントに対してもイベント境界を決定する
    def event_search(self, sections, event_index, threshold, results):
        event_features = []
        if type(threshold) == np.float64:
            threshold = [threshold for _ in range(len(event_index))]

        for j, thr in zip(event_index, threshold):
            event_feature = self.boundary_search(results[j], j)
            if event_feature is not None:
                event_features_before = self.recursive_event_search([sections[j][0], event_feature[4]], thr, j)
                event_features_after = self.recursive_event_search([event_feature[5]+1, sections[j][1]], thr, j)
                event_features.extend(event_features_before + [event_feature] + event_features_after)
                
        return event_features
    
    #区間に存在するイベントを再帰的に探索する
    def recursive_event_search(self, search_range, threshold, j):
        # 探索範囲の平均と最小値の差を計算し、閾値チェック
        if search_range[1] - search_range[0] > 0:
            section_data = self.data[search_range[0]:search_range[1]]
            mean_value = np.mean(section_data)
            min_value = np.min(section_data)
        else:
            return []
        
        if mean_value - min_value <= threshold:
            # 閾値を超えない場合、空の配列を返す
            return []

        # 閾値を超える場合、イベント境界を決定
        min_index = np.argmin(section_data) + search_range[0]
        Pj = self.data[min_index]
        Bj = mean_value
        Aj = Bj - Pj
        
        # `boundary_search`でイベントの始点と終点を特定
        event_feature = self.boundary_search([Pj, min_index, Bj, Aj, False], j, section_data)

        # イベント境界が見つからない場合、空の配列を返す
        if event_feature is None:
            return []

        # 境界の前方・後方について再帰的に探索
        event_start = event_feature[4]
        event_end = event_feature[5]
        
        # 前方の探索
        events_before = self.recursive_event_search((search_range[0], event_start), threshold, j)
        
        # 後方の探索
        events_after = self.recursive_event_search((event_end+1, search_range[1]), threshold, j)
        # 前方のイベントリスト、現在のイベント、後方のイベントリストを連結して返す
        return events_before + [event_feature] + events_after

    #イベントの始点と終点を探す
    def boundary_search(self, result_j, j, section_data=None):
        Pj, Kj, Bj, Aj, _ = result_j
        event_start = Kj
        event_end = Kj

        for t in range(event_start, -1, -1):
            if self.data[t] >= Bj:
                event_start = t
                break
        else:
            if j != 0:
                print("Error: Start point was not found")
            return None
        
        for t in range(event_end, len(self.data)):
            if self.data[t] >= Bj:
                event_end = t
                break
        else:
            if j != self.final_event_contain_section:
                print("Error: End point was not found")
            return None
        
        return [Pj, Kj, Bj, Aj, event_start, event_end]
    
    
    #イベントの特徴量を計算する
    def calculate_eventdata(self, event_features):
        eventdata = []
        for i in range(len(event_features)):
            Pj, Kj, Bj, Aj, event_start, event_end = event_features[i]
            Pj = Pj
            Bj = Bj
            Aj = Aj
            Maxnormblock = Aj / Bj
            if Maxnormblock < 0:
                Maxnormblock = -Maxnormblock
            Kj = Kj / self.sampling_rate * 1000000
            TSj = event_start * self.point_interval
            TEj = event_end * self.point_interval
            Lj = TEj - TSj
            AveBlock = Bj - (np.mean(self.data[event_start:event_end+1]))
            if AveBlock < 0:
                AveBlock = -AveBlock
            AveNormBlock = AveBlock / Bj
            eventdata.append((Lj, Bj, Pj, Kj, Aj, Maxnormblock, AveBlock, AveNormBlock, TSj, TEj, event_start, event_end))
        return eventdata

    #result.csvの保存
    def save_result(self, eventdata, threshold):
        #outputフォルダの作成
        os.makedirs(self.output_path, exist_ok=True)
        # if os.path.exists(self.output_path):
        #     shutil.rmtree(self.output_path)
        # os.mkdir(self.output_path)

        # df = pd.DataFrame(eventdata, columns=[
        #     'Duration[us]', 'Baseline[pA]', 'Peak Current value[pA]', 'PeakTime[us]', 
        #     'Max Blockage[pA]', 'Normalized Max Blockage', 'Average Blockage[pA]', 'Normlized Average Blockage', 
        #     'StartTime[us]', 'EndTime[us]', 'StartIndex', 'EndIndex'
        # ])
        # output_file = os.path.join(self.output_path, f'{self.output_file_prefix}_result.csv')
        # df.to_csv(output_file, index=False)

        with open(os.path.join(self.output_path, f'{self.output_file_prefix}_log.txt'), mode='w') as f:
            if self.control_trainer[0] > 0.01:
                f.write("Fitting control data succeeded\n")
                f.write(f"Threshold: {threshold[0]}\n")
            elif type(threshold) == np.float64:
                f.write(f"Threshold: {threshold}\n")
            elif threshold == []:
                f.write("Events were not detected\n")
            else:
                f.write(f"Threshold: {threshold[0]}\n")
            f.write(f"Number of events: {len(eventdata)}\n")
            f.write(f"Frequency: {len(eventdata) / (len(self.data) / self.sampling_rate)} /s\n")
    
    #イベントの電流データを保存
    def save_currents(self, eventdata):
        # events_dir = os.path.join(self.output_path, 'EWMAuto_events') # 分かりやすいように変数に格納
        events_dir = os.path.join(self.output_path, f'{self.output_file_prefix}_events') # 変更後
        if os.path.exists(events_dir):
            shutil.rmtree(events_dir)
        os.mkdir(events_dir)
    
        limit = len(eventdata)
        for i in range(limit):
            event = eventdata[i]
            current = {
                'Amplitude[pA]': self.data[event[10]:event[11]+1],
                'Time[us]': np.arange(event[10], event[11]+1) * self.point_interval,
            }
            df_event = pd.DataFrame(current)
            # df_event.to_csv(os.path.join(self.output_path, 'EWMAuto_events', f'event_{i}.csv'), index=False) # 変更前
            df_event.to_csv(os.path.join(events_dir, f'event_{i}.csv'), index=False) # 変更後
        #もともとこれ

     #   for i in range(len(eventdata)):
      #      event = eventdata[i]
       #     current = {
        #        'Amplitude[pA]': self.data[event[10]:event[11]+1],
         #       'Time[us]': np.arange(event[10], event[11]+1) * self.point_interval,
          #  }
           # df_event = pd.DataFrame(current)
            #df_event.to_csv(os.path.join(self.output_path, 'EWMAuto_events', f'event_{i}.csv'), index=False)  

            
#田村が追加したところ
        #EWMAuto_result と EWMAuto_events/event_i.csv から
        #13次元特徴量を計算して EWMAuto_features.csv で保存
    def save_features_csv(self, eventdata):  # 引数に eventdata を追加
        features =[]
        
        for i in range(len(eventdata)):
            # eventdata から直接情報を取得
            Lj, Bj, Pj, Kj, Aj, Maxnormblock, AveBlock, AveNormBlock, TSj, TEj, event_start, event_end = eventdata[i]

            # CSVから読み込まずに、メモリ上の波形データを直接スライスして取得
            amplitude = self.data[event_start:event_end+1]
            time = np.arange(event_start, event_end+1) * self.point_interval
            baseline = Bj

            # Peak Position 
            if Lj == 0:
                continue
            peak_position = (Kj - TSj) / Lj

            # area / skewness / kurtosis 
            current = np.abs(baseline - amplitude)
            area = np.sum(current)

            if area == 0:
                continue

            freq = current / area
            t = time - time.min()

            mean = np.sum(freq * t)
            std_t = np.sqrt(np.sum(freq * (t - mean) ** 2))

            if std_t == 0:
                continue

            skewness = np.sum(freq * ((t - mean) / std_t) ** 3)
            kurtosis = np.sum(freq * ((t - mean) / std_t) ** 4)

            # slope 
            min_value = amplitude.min()
            amp = baseline - min_value

            a90 = baseline - 0.9 * amp
            a10 = baseline - 0.1 * amp

            # 波形から該当する箇所のインデックスを高速に取得
            under90_idx = np.where(amplitude <= a90)[0]
            under10_idx = np.where(amplitude <= a10)[0]

            if len(under90_idx) == 0 or len(under10_idx) == 0:
                continue

            idx90_first, idx90_last = under90_idx[0], under90_idx[-1]
            idx10_first, idx10_last = under10_idx[0], under10_idx[-1]

            dt_left = time[idx90_first] - time[idx10_first]
            dt_right = time[idx10_last] - time[idx90_last]

            # ゼロ除算の回避
            if dt_left == 0 or dt_right == 0:
                continue

            leftslope = (amplitude[idx90_first] - amplitude[idx10_first]) / dt_left
            rightslope = (amplitude[idx10_last] - amplitude[idx90_last]) / dt_right

            # FWHM
            half = baseline - 0.5 * amp
            under_half_idx = np.where(amplitude <= half)[0]

            if len(under_half_idx) == 0:
                continue

            fwhm = time[under_half_idx[-1]] - time[under_half_idx[0]]

            # リストに計算結果を追加
            features.append([
                Lj,             # Duration[us]
                Aj,             # Max Blockage[pA]
                Maxnormblock,   # Normalized Max Blockage
                AveBlock,       # Average Blockage[pA]
                AveNormBlock,   # Normlized Average Blockage
                peak_position,
                area,
                leftslope,
                rightslope,
                np.std(amplitude),
                kurtosis,
                skewness,
                fwhm
            ])

        # カラム名の定義
        columns = [
            "Duration[us]",
            "Max Blockage[pA]",
            "Normalized Max Blockage",
            "Average Blockage[pA]",
            "Normlized Average Blockage",
            "Peak Position",
            "area[us*pA]",
            "leftslope[pA/us]",
            "rightslope[pA/us]",
            "std[pA]",
            "kurtosis",
            "skewness",
            "fwhm[us]"
        ]

        df_features = pd.DataFrame(features, columns=columns)
        df_features = df_features.dropna(how="any")

        # featureファイルの名前は接頭辞をつける
        output_csv = os.path.join(self.output_path, f"{self.output_file_prefix}_features.csv")
        df_features.to_csv(output_csv, index=False)

        if self.log:
            print(f"[OK] Feature CSV saved: {output_csv}")
    