from sklearn.pipeline import Pipeline


class MLPipeline(Pipeline):
    """
    this is a general machine learning pipeline used to make the Auto ML pipeline
    it's an extension of the Pipeline class created by sklearn
    """
    def __init__(self, name="MLPipeline", steps=[]):

        super(MLPipeline, self).__init__(steps)
        self.name = name
