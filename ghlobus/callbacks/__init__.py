# ghlobus/callbacks/__init__.py
# Writes frame embeddings, context vectors, attention weights
from .CnnVectorWriter import CnnVectorWriter
# Callbacks for GA model
from .GaVideoPredictionWriter import GaVideoPredictionWriter
from .GaExamPredictionWriter import GaExamPredictionWriter
# Callbacks for FP, TWIN-BAA models
from .ClassificationVideoPredictionWriter import ClassificationVideoPredictionWriter
from .ClassificationExamPredictionWriter import ClassificationExamPredictionWriter
# Callbacks for EFW model
from .EfwExamPredictionWriter import EfwExamPredictionWriter
from .EfwVideoPredictionWriter import EfwVideoPredictionWriter
# Callback for TWIN-iMIL model
from .TwinExamPredictionWriter import TwinExamPredictionWriter
