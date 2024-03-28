

from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator


def test_pattern_generators():
    error_model = ErrorModel.from_json_file('error_models/models/conv_gemm.json')
    print(error_model)

    fault_gen = FaultGenerator(error_model)
    faults = fault_gen.generate_fault_list(100, (1,3,416,416))

    print(faults)

if __name__ == '__main__':
     test_pattern_generators()