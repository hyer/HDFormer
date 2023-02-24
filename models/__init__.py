def get_model(skeleton, cfg, logger):
    logger.info('Using {} network'.format(cfg.arch))
    if cfg.arch == 'model_vertex':
        from models.vertex_model import Model
        model = Model(skeleton=skeleton, cfg=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model
