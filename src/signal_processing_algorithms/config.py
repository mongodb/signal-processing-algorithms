import structlog


class Configuration(object):
    _logger = structlog.get_logger()

    @staticmethod
    def configure(logger):
        Configuration._logger = logger

    @staticmethod
    def get_logger():
        return Configuration._logger
