import analysis_confs


class AnalysisConfigurator:
    def __init__(self, graph):
        self.graph = graph
        self.min_variant_freq = analysis_confs.min_variant_freq[self.graph]

        self.num_clusters = analysis_confs.num_clusters[self.graph]
        self.cluster_min_variant_length = analysis_confs.cluster_min_variant_length[self.graph]
        self.manual_clusters = analysis_confs.manual_clusters[self.graph]
        self.cluster_include_remainder = analysis_confs.cluster_include_remainder[self.graph]
        self.leftover_cluster = analysis_confs.leftover_cluster[self.graph]

        self.clustering_setting = f"V{self.min_variant_freq}_C{self.num_clusters}_" \
                                               f"L{self.cluster_min_variant_length}"

        if self.manual_clusters is not "":
            self.clustering_setting += "_manual"

        if self.cluster_include_remainder:
            self.clustering_setting += "_Rinc"
        else:
            self.clustering_setting += "_Rexc"


    def get_analysis_directory(self):
        analysis_directory = f"F:\\analysis_output\\{self.graph}\\{self.clustering_setting}"
        # analysis_directory = f"output\\{self.graph}\\{self.clustering_setting}"
        return analysis_directory

    def get_min_variant_freq(self):
        return self.min_variant_freq

    def get_num_clusters(self):
        return self.num_clusters

    def get_cluster_min_variant_length(self):
        return self.cluster_min_variant_length

    def get_manual_clusters(self):
        return self.manual_clusters

    def get_cluster_include_remainder(self):
        return self.cluster_include_remainder

    def get_leftover_cluster(self):
        return self.leftover_cluster

    def get_clustering_setting(self):
        return self.clustering_setting
