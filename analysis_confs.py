#####################################################
############# EXPLANATION OF SETTINGS ###############
#####################################################

# SUBSET FILTER
min_variant_freq = {}

# CLUSTER SETTINGS
num_clusters = {}

cluster_min_variant_length = {}
manual_clusters = {}
cluster_include_remainder = {}
leftover_cluster = {}

for graph in ["bpic2017_susp_res"]:
    min_variant_freq[graph] = 10
    num_clusters[graph] = 21
    cluster_min_variant_length[graph] = 2
    manual_clusters[graph] = {"m01": [1, 25, 107, 138, 176, 185, 189, 192],
                              "m02": [2, 57, 87, 129, 179, 213, 216, 217],
                              "m03": [3, 33, 124, 139, 166, 194, 197, 211, 222, 238],
                              "m04": [7, 9, 45, 65, 150, 160, 225, 228, 233, 240],
                              "m05": [20],
                              "m06": [50, 78, 144, 196],
                              "m07": [56, 195, 268]}
    cluster_include_remainder[graph] = True
    leftover_cluster[graph] = "T05"

