import argparse
import datetime as dtm
import logging
from .config.model_config import ModelConfig
from .utils import logs

PKG_NAME = "odonto"

#Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_run", required=False, help="Data/Hora de execução do experimento.", type=str, default=dtm.datetime.now().strftime("%Y%m%d%H%M%S"))
args = parser.parse_args()
config = ModelConfig(experiment_run=args.experiment_run)

# Configure logger for use in package
logger = logging.getLogger(__name__)
level = logging.getLevelName(config.logging.level)
logger.setLevel(level)
logger.addHandler(logs.get_console_handler())
# logger.propagate = False

logger.info(f"Carregando configurações do branch {config.environment.value}")