# lower dimensional problem for SMAC
# from smac_cs_40 import cfg2funcparams_40  # noqa
# from smac_cs_40 import cs_40  # NOQA

# from smac_cs_single import cfg2funcparams_single  # NOQA
# from smac_cs_single import cs_single  # NOQA

# resnet50
from space_utils.smac_cs_resnet50_multiple import cfg2funcparams as cfg2funcparams_resnet50  # noqa
from space_utils.smac_cs_resnet50_multiple import get_cs as get_cs_resnet50  # NOQA

# vgg16
from space_utils.smac_cs_vgg16_multiple import cfg2funcparams as cfg2funcparams_vgg16  # NOQA
from space_utils.smac_cs_vgg16_multiple import get_cs as get_cs_vgg16 # noqa

# resnet56
from space_utils.smac_cs_resnet56 import cfg2funcparams as cfg2funcparams_resnet56
from space_utils.smac_cs_resnet56 import get_cs as get_cs_resnet56
