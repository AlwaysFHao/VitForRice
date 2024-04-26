import logging


def setting_logging(log_name):
    """
    设置日志
    :param log_name: 日志名
    :return: 可用日志
    """
    # 第一步：创建日志器对象，默认等级为warning
    logger = logging.getLogger(log_name)
    logging.basicConfig(level="INFO")

    # 第二步：创建控制台日志处理器
    console_handler = logging.StreamHandler()

    # 第三步：设置控制台日志的输出级别,需要日志器也设置日志级别为info；----根据两个地方的等级进行对比，取日志器的级别
    console_handler.setLevel(level="WARNING")

    # 第四步：设置控制台日志的输出格式
    console_fmt = "%(name)s--->%(asctime)s--->%(message)s--->%(lineno)d"
    fmt1 = logging.Formatter(fmt=console_fmt)
    console_handler.setFormatter(fmt=fmt1)

    # 第五步：将控制台日志器，添加进日志器对象中
    logger.addHandler(console_handler)

    return logger
