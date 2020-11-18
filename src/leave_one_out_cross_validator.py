class LeaveOneOutCrossValidator:
    def __init__(self, classifier, json_data):
        self.classifier = classifier
        self.json_data = json_data

    def accuracy(self):
        return sum(1 if self.classifier(i) else 0 for i in self.json_data)/len(self.json_data)
