import os


import plotly.express as px
import plotly.graph_objects as go

class LoggingTracker:
    """ Log informations and plot the every graph.
    """
    def __init__(self, logger, writer, config, title):
        self.writer = writer
        self.logger = logger

        self.config = config
        self.title = title
