
import os
import re
import functools as funct

precisions = ['int8', 'int16', 'int32']

FULL_UNITS = [
    'top.conv.cdma',
    'top.conv.cbuf',
    'top.conv.cbuf.datbuf',
    'top.conv.cbuf.wtbuf',
    'top.conv.csb',
    'top.conv.csc',
    'top.conv.dl',
    'top.conv.wl',
    'top.conv.cmac',
    'top.conv.cacc',
    'top.conv.dbuf',
    'top.sdp'
]

UNITS = [
    'top.conv.csb',
    'top.conv.csc',
    'top.conv.dl',
    'top.conv.wl',
    'top.conv.cdma',
    'top.conv.cbuf',
    'top.conv.cmac',
    'top.conv.cacc',
    'top.sdp',
    'top',
]

UNIT_RENAME = {
    "top.conv.csb":  "CSB",
    "top.conv.csc":  "CSC",
    "top.conv.dl":   "DL",
    "top.conv.wl":   "WL",
    "top.conv.cdma": "CDMA",
    "top.conv.cbuf": "CBUF",
    "top.conv.cmac": "CMAC",
    "top.conv.cacc": "CACC",
    "top.sdp":       "SDP",
    "top":           "TOTAL",
}

BENCHMARK_RENAME = {
    "alexnet_cifar10": "AlexNet",
    "res9_cifar10": "ResNet9",
    "res18_cifar100": "ResNet18",
    "res50_cifar10": "ResNet50",
    "vgg11_cifar10": "VGG11",
    "lenet_cifar10": "LeNet",
    "deeplabv3_oxfordpet": "DeepLabV3",
    "mobilenetv2_gtsrb": "MobileNetV2",
    "mobilenetv2-large_gtsrb": "MobileNetV2-L",
    "yolov11_coco": "YOLOv11",
}

def benchmark_reformat(name: str) -> str:
    return BENCHMARK_RENAME.get(name, name)

def config_reformat(configname: str | list[str]) -> str | list[str]:
    pattern = re.compile('^nv_\d+x\d+', re.IGNORECASE)
    if isinstance(configname, list):
        return [ re.match(pattern, c).group(0).replace('_', '') for c in configname ]
    else:
        return re.match(pattern, configname).group(0).replace('_', '')


def sort_configs(configs: list[str], policy: str = 'base', precision_suffix: bool = False) -> list[str]:
    if policy == 'k-group':
        def make_sort_key(sep="_", suffix_present=True):
            ck_re = re.compile(r'(\d+)x(\d+)')
            p_re  = re.compile(r'(\d+)$')

            def key(s):
                fields = s.split(sep)

                # extract c, k from second field
                # c, k = map(int, ck_re.search(fields[1]).groups())
                c, k = map(int, ck_re.search(fields[0]).groups())

                # extract p from last field (optional)
                if suffix_present:
                    m = p_re.search(fields[-1])
                    p = int(m.group(1)) if m else -1
                else:
                    p = -1

                return (p, k, c)
            return key

        return sorted(configs, key=make_sort_key("_", precision_suffix))

    else:
        return sorted(configs, key=lambda x: tuple(map(int, re.findall(r'\d+', x)[:5])))

def sort_nets(nets: list[str], outdir: str, precision: str = 'int8') -> list[str]:
    return sorted(
        nets,
        key=lambda b: sum(1 for e in os.scandir(os.path.join(outdir, 'benchmarks', b, 'layers', precision)) if os.path.isdir(e)))

def extract_configs(filelist: list[str]) -> list[str]:
    return [ str(os.path.basename(f).removesuffix('.yaml')) for f in filelist ]

def detach_precision(config: str) -> str:
    precision = config.split('_')[-1]
    id        = config.removesuffix(f'_{precision}')
    return id, precision
