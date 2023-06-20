import graph_confs


class GraphConfigurator:
    def __init__(self, graph_name):
        self.graph_name = graph_name
        self.filename = graph_confs.filename[self.graph_name]
        self.name_data_set = graph_confs.name_data_set[self.graph_name]
        self.column_names = graph_confs.column_names[self.graph_name]
        self.timestamp_label = graph_confs.timestamp_label[self.graph_name]
        # self.case_attr_labels = graph_confs.case_attr_labels[self.graph_name]
        self.separator = graph_confs.separator[self.graph_name]
        self.timestamp_format = graph_confs.timestamp_format[self.graph_name]

        self.password = graph_confs.password[self.graph_name]
        self.entity_labels = graph_confs.entity_labels[self.graph_name]
        self.action_lifecycle_labels = graph_confs.action_lifecycle_labels[self.graph_name]

    def get_filename(self):
        return self.filename

    def get_name_data_set(self):
        return self.name_data_set

    def get_column_names(self):
        return self.column_names

    def get_timestamp_label(self):
        return self.timestamp_label

    def case_attr_labels(self):
        return self.case_attr_labels

    def get_separator(self):
        return self.separator

    def get_timestamp_format(self):
        return self.timestamp_format

    def get_password(self):
        return self.password

    def get_entity_labels(self):
        return self.entity_labels

    def get_action_lifecycle_labels(self):
        return self.action_lifecycle_labels
