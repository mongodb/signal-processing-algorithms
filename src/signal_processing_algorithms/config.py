# -*- coding: utf-8 -*-
"""Provide a common configuration interface for signal processing."""
import structlog


class Configuration(object):
    """This class encapsulates the common configuration instances ."""

    _logger = structlog.get_logger()

    @staticmethod
    def configure(logger):
        """Set the configuration.

        Currently we only support configuring the logger instance.
        :param logger: the instance to use to log messages.
        """
        Configuration._logger = logger

    @staticmethod
    def get_logger():
        """
        Get the configured the logger instance.

        :return: The logger instance.
        """
        return Configuration._logger
