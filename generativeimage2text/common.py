from tqdm import tqdm
import json
from PIL import Image
import base64
import random
import os
import argparse
import sys
import os.path as op
import yaml
import logging
from pprint import pformat


class Config(object):
    def __init__(self, default, overwrite):
        self.default = default
        self.overwrite = overwrite

    def get(self, k):
        if dict_has_path(self.default, k):
            base = dict_get_path_value(self.default, k)
        else:
            base = None
        if dict_has_path(self.overwrite, k):
            over = dict_get_path_value(self.overwrite, k)
            if isinstance(base, dict):
                assert isinstance(over, dict)
                base.update(over)
            else:
                base = over
        return base

    def __getattr__(self, k):
        return self.get(k)

    def __copy__(self):
        return Config(self.default, self.overwrite)

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return Config(deepcopy(self.default), deepcopy(self.overwrite))

    def get_dict(self):
        import copy
        default = copy.deepcopy(self.default)
        for p in get_all_path(self.overwrite, with_list=False):
            v = dict_get_path_value(self.overwrite, p)
            dict_update_path_value(default, p, v)
        return default

def dict_remove_path(d, p):
    ps = p.split('$')
    assert len(ps) > 0
    cur_dict = d
    need_delete = ()
    while True:
        if len(ps) == 1:
            if len(need_delete) > 0 and len(cur_dict) == 1:
                del need_delete[0][need_delete[1]]
            else:
                del cur_dict[ps[0]]
            return
        else:
            if len(cur_dict) == 1:
                if len(need_delete) == 0:
                    need_delete = (cur_dict, ps[0])
            else:
                need_delete = (cur_dict, ps[0])
            cur_dict = cur_dict[ps[0]]
            ps = ps[1:]

def dict_has_path(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, dict) and k in cur_dict:
                cur_dict = cur_dict[k]
                ps = ps[1:]
            elif isinstance(cur_dict, list):
                try:
                    k = int(k)
                except:
                    return False
                cur_dict = cur_dict[k]
                ps = ps[1:]
            else:
                return False
        else:
            return True


def dict_update_nested_dict(a, b, overwrite=True):
    for k, v in b.items():
        if k not in a:
            dict_update_path_value(a, k, v)
        else:
            if isinstance(dict_get_path_value(a, k), dict) and isinstance(v, dict):
                dict_update_nested_dict(dict_get_path_value(a, k), v, overwrite)
            else:
                if overwrite:
                    dict_update_path_value(a, k, v)

def get_mpi_rank():
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))

def get_mpi_local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))

def get_mpi_size():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))

def dict_ensure_path_key_converted(a):
    for k in list(a.keys()):
        v = a[k]
        if '$' in k:
            parts = k.split('$')
            x = {}
            x_curr = x
            for p in parts[:-1]:
                x_curr[p] = {}
                x_curr = x_curr[p]
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)
            x_curr[parts[-1]] = v
            dict_update_nested_dict(a, x)
            del a[k]
        else:
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)

def dict_update_path_value(d, p, v):
    ps = p.split('$')
    while True:
        if len(ps) == 1:
            d[ps[0]] = v
            break
        else:
            if ps[0] not in d:
                d[ps[0]] = {}
            d = d[ps[0]]
            ps = ps[1:]

def dict_parse_key(k, with_type):
    if with_type:
        if k[0] == 'i':
            return int(k[1:])
        else:
            return k[1:]
    return k

def dict_get_path_value(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, (tuple, list)):
                cur_dict = cur_dict[int(k)]
            else:
                cur_dict = cur_dict[k]
            ps = ps[1:]
        else:
            return cur_dict

def releaseLock(locked_file_descriptor):
    locked_file_descriptor.close()

def print_trace():
    import traceback
    traceback.print_exc()

def hash_sha1(s):
    import hashlib
    if type(s) is not str:
        s = pformat(s)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def write_to_file(contxt, file_name, append=False):
    p = os.path.dirname(file_name)
    ensure_directory(p)
    if type(contxt) is str:
        contxt = contxt.encode()
    flag = 'wb'
    if append:
        flag = 'ab'
    with open(file_name, flag) as fp:
        fp.write(contxt)

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise

def pilimg_from_base64(imagestring):
    try:
        import io
        jpgbytestring = base64.b64decode(imagestring)
        image = Image.open(io.BytesIO(jpgbytestring))
        image = image.convert('RGB')
        return image
    except:
        return None

def json_dump(obj):
    # order the keys so that each operation is deterministic though it might be
    # slower
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))

def acquireLock(lock_f='/tmp/lockfile.LOCK'):
    import fcntl
    ensure_directory(op.dirname(lock_f))
    locked_file_descriptor = open(lock_f, 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor

def get_user_name():
    import getpass
    return getpass.getuser()

def limited_retry_agent(num, func, *args, **kwargs):
    for i in range(num):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning('fails with \n{}: tried {}/{}-th time'.format(
                e,
                i + 1,
                num,
            ))
            import time
            print_trace()
            if i == num - 1:
                raise
            t = random.random() * 5
            time.sleep(t)

def exclusive_open_to_read(fname, mode='r'):
    disable_lock = os.environ.get('QD_DISABLE_EXCLUSIVE_READ_BY_LOCK')
    if disable_lock is not None:
        disable_lock = int(disable_lock)
    if not disable_lock:
        user_name = get_user_name()
        lock_fd = acquireLock(op.join('/tmp',
            '{}_lock_{}'.format(user_name, hash_sha1(fname))))
    #try:
    # in AML, it could fail with Input/Output error. If it fails, we will
    # use azcopy as a fall back solution for reading
    fp = limited_retry_agent(10, open, fname, mode)
    if not disable_lock:
        releaseLock(lock_fd)
    return fp

def read_to_buffer(file_name):
    with open(file_name, 'rb') as fp:
        all_line = fp.read()
    return all_line

def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result

def init_logging():
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(process)d:%(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
    ch.setFormatter(logger_fmt)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(ch)
    root.setLevel(logging.INFO)

def load_from_yaml_str(s):
    return yaml.load(s, Loader=yaml.UnsafeLoader)

def get_all_path(d, with_type=False, leaf_only=True, with_list=True):
    assert not with_type, 'will not support'
    all_path = []

    if isinstance(d, dict):
        for k, v in d.items():
            all_sub_path = get_all_path(
                v, with_type, leaf_only=leaf_only, with_list=with_list)
            all_path.extend([k + '$' + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append(k)
    elif (isinstance(d, tuple) or isinstance(d, list)) and with_list:
        for i, _v in enumerate(d):
            all_sub_path = get_all_path(
                _v, with_type,
                leaf_only=leaf_only,
                with_list=with_list,
            )
            all_path.extend(['{}$'.format(i) + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append('{}'.format(i))
    return all_path

def load_from_yaml_file(file_name):
    # do not use QDFile.open as QDFile.open depends on this function
    with exclusive_open_to_read(file_name, 'r') as fp:
    #with open(file_name, 'r') as fp:
        data = load_from_yaml_str(fp)
    while isinstance(data, dict) and '_base_' in data:
        b = op.join(op.dirname(file_name), data['_base_'])
        result = load_from_yaml_file(b)
        assert isinstance(result, dict)
        del data['_base_']
        all_key = get_all_path(data, with_list=False)
        for k in all_key:
            v = dict_get_path_value(data, k)
            dict_update_path_value(result, k, v)
        data = result
    return data

def parse_general_args():
    parser = argparse.ArgumentParser(description='General Parser')
    parser.add_argument('-c', '--config_file', help='config file',
            type=str)
    parser.add_argument('-p', '--param', help='parameter string, yaml format',
            type=str)
    parser.add_argument('-bp', '--base64_param', help='base64 encoded yaml format',
            type=str)
    args = parser.parse_args()
    kwargs =  {}
    if args.config_file:
        logging.info('loading parameter from {}'.format(args.config_file))
        configs = load_from_yaml_file(args.config_file)
        for k in configs:
            kwargs[k] = configs[k]
    if args.base64_param:
        configs = load_from_yaml_str(base64.b64decode(args.base64_param))
        for k in configs:
            if k not in kwargs:
                kwargs[k] = configs[k]
            elif kwargs[k] == configs[k]:
                continue
            else:
                logging.info('overwriting {} to {} for {}'.format(kwargs[k],
                    configs[k], k))
                kwargs[k] = configs[k]
    if args.param:
        configs = load_from_yaml_str(args.param)
        dict_ensure_path_key_converted(configs)
        for k in configs:
            if k not in kwargs:
                kwargs[k] = configs[k]
            elif kwargs[k] == configs[k]:
                continue
            else:
                logging.info('overwriting {} to {} for {}'.format(kwargs[k],
                    configs[k], k))
                kwargs[k] = configs[k]
    return kwargs

def qd_tqdm(*args, **kwargs):
    desc = kwargs.get('desc', '')
    import inspect
    frame = inspect.currentframe()
    frames = inspect.getouterframes(frame)
    frame = frames[1].frame
    line_number = frame.f_lineno
    fname = op.basename(frame.f_code.co_filename)
    message = '{}:{}'.format(fname, line_number)

    if 'desc' in kwargs:
        kwargs['desc'] = message + ' ' + desc
    else:
        kwargs['desc'] = message

    if 'mininterval' not in kwargs:
        # every 2 secons; default is 0.1 second which is too frequent
        kwargs['mininterval'] = 2

    return tqdm(*args, **kwargs)

