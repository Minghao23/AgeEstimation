from multiprocessing import Pool, TimeoutError
import age_gender_project.demo
import logging
import time


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def process_work(photo_path, config):
    import os
    logger.info('pid is %s' % os.getpid())
    logger.info('photo is %s' % photo_path)

    # default config
    depth = 16
    k = 8
    margin = 0.4
    balance = 0.5

    # load config
    if config is not None:
        if 'depth' in config:
            depth = config['depth']
        if 'width' in config:
            k = config['width']

        if 'margin' in config:
            margin = config['margin']
        if 'balance' in config:
            balance = config['balance']

    res = age_gender_project.demo.my_test(photo_path, depth, k, margin, balance)

    return res


pool = Pool()


def detect_gender_age(photo_path, config=None):
    logger.info('Detecting %s' % photo_path)
    timeout = 60
    p = pool.apply_async(process_work, args=(photo_path, config))
    try:
        results = p.get(timeout)
    except TimeoutError:
        logger.error('Timeout. Running time is longer than %d for detection' % timeout)
        return {'message': 'It took too long to complete detection.', 'successful': False, 'code': 2}
    except Exception:
        logger.error('Something wrong occurred while detecting in subprocess')
        return {'message': 'Something wrong occurred while detecting in subprocess', 'successful': False, 'code': 3}

    response_dict = {'successful': True, 'results': results}

    return response_dict
