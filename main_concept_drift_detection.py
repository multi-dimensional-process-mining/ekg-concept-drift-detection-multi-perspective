import os

from GraphConfigurator import GraphConfigurator
from AnalysisConfigurator import AnalysisConfigurator
from EventGraph import EventGraph
from concept_drift_detection import concept_drift_analysis

# initialize graph and analysis settings
graph = "bpic2017_susp_res"
gc = GraphConfigurator(graph)
ac = AnalysisConfigurator(graph)
eg = EventGraph(gc.get_password(), gc.get_entity_labels())

# create global analysis directory
analysis_directory = os.path.join(ac.get_analysis_directory(), "concept_drift_detection")
os.makedirs(analysis_directory, exist_ok=True)

window_sizes = [1]
penalties = [3]

step_process_level_drift_detection = True
process_drift_feature_sets = {"task_relative": ["count_per_task_relative"],
                              "task_variant_relative": ["count_per_task_variant_relative"],
                              "activity_relative": ["count_per_activity_relative"],
                              "activity_handover_case_relative": ["count_per_activity_handover_case_relative"]}

step_actor_drift_detection = False
actor_drift_feature_sets = {"task_handover_actor_relative": ["count_per_task_handover_actor_relative"]}
actors = eg.query_actor_list(min_freq=500)


step_collab_drift_detection = False
collab_pairs = eg.query_collab_list(min_freq=300)


step_eval_actor_vs_process_drift = False
step_eval_collab_vs_process_drift = False

step_calculate_magnitude = False

step_calculate_change_magnitude_percentiles = False

step_detailed_change_signal_analysis = False
cp_task_dict = {1: ["m04", "T03", "T07"],
                2: ["m01", "m03", "m04", "T01", "T02", "T11"],
                3: ["m02", "m04", "m05"],
                4: ["m05", "T11"]}
cp_variant_dict = {2: [4, 6, 8, 13],
                   3: [4, 27, 12, 43]}

if step_process_level_drift_detection:
    print("START PROCESS DRIFT DETECTION")
    concept_drift_analysis.detect_process_level_drift(graph=graph, window_sizes=window_sizes, penalties=penalties,
                                                      feature_sets=process_drift_feature_sets,
                                                      analysis_directory=analysis_directory, event_graph=eg,
                                                      exclude_cluster=ac.get_leftover_cluster(), plot_drift=False)

if step_actor_drift_detection:
    print("START ACTOR DRIFT DETECTION")
    concept_drift_analysis.detect_actor_drift(graph=graph, window_sizes=window_sizes, penalties=penalties,
                                              feature_sets=actor_drift_feature_sets, actor_list=actors,
                                              analysis_directory=analysis_directory, event_graph=eg,
                                              exclude_cluster=ac.get_leftover_cluster(), plot_drift=True)

if step_collab_drift_detection:
    print("START collab DRIFT DETECTION")
    concept_drift_analysis.detect_collab_drift(graph=graph, window_sizes=window_sizes, penalties=penalties,
                                               detailed_analysis=True, collab_list=collab_pairs,
                                               analysis_directory=analysis_directory, event_graph=eg,
                                               exclude_cluster=ac.get_leftover_cluster(), plot_drift=True)

if step_eval_actor_vs_process_drift:
    print("START EVALUATION ACTOR DRIFT VS PROCESS LEVEL DRIFT")
    concept_drift_analysis.eval_subgroup_vs_process_drift(window_size=1, pen_process=3, pen_subgroup=4,
                                                          subgroup_type="actor",
                                                          feature_set_name_subgroup="task_handover_actor_relative",
                                                          feature_set_name_process_level="task_relative",
                                                          analysis_directory=analysis_directory)

if step_eval_collab_vs_process_drift:
    print("START EVALUATION COLLAB DRIFT VS PROCESS LEVEL DRIFT")
    concept_drift_analysis.eval_subgroup_vs_process_drift(window_size=1, pen_process=3, pen_subgroup=4,
                                                          subgroup_type="collab",
                                                          feature_set_name_subgroup="task_handovers_case",
                                                          feature_set_name_process_level="task_relative",
                                                          analysis_directory=analysis_directory)

if step_calculate_magnitude:
    print("START CALCULATING SIGNAL MAGNITUDE")
    concept_drift_analysis.calculate_signal_magnitude(graph=graph, window_size=1, penalty=4,
                                                      feature_sets=process_drift_feature_sets,
                                                      analysis_directory=analysis_directory, event_graph=eg,
                                                      exclude_cluster=ac.get_leftover_cluster())

if step_calculate_change_magnitude_percentiles:
    print("START CALCULATING CHANGE MAGNITUDE PERCENTILES")
    concept_drift_analysis.calculate_change_magnitude_percentiles(graph=graph, window_size=1, penalty=4,
                                                                  feature_sets=process_drift_feature_sets,
                                                                  analysis_directory=analysis_directory, event_graph=eg,
                                                                  exclude_cluster=ac.get_leftover_cluster())

if step_detailed_change_signal_analysis:
    print("START DETAILED CHANGE ANALYSIS")
    concept_drift_analysis.compare_tasks_vs_activity_actvity_pair(graph=graph, penalty=3,
                                                                  analysis_directory=analysis_directory,
                                                                  event_graph=eg, cp_task_dict=cp_task_dict)
    concept_drift_analysis.compare_variant_vs_activity_actvity_pair(graph=graph, penalty=3,
                                                                    analysis_directory=analysis_directory,
                                                                    event_graph=eg, cp_variant_dict=cp_variant_dict)
