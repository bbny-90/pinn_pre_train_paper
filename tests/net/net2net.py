import inspect

def test1():
    import numpy as np
    import torch
    from models.nueral_net_pt import MLP
    bsz, dim = 10, 3
    params = {'in_dim':dim,
              'out_dim':2,
              'hid_dim':5,
              'num_hid_layer':3,
              'act_type':'relu',
             }
    with torch.no_grad():
        x = torch.rand(bsz, dim)
        model = MLP(params)
        y = model(x).numpy()
        model_widen = model.widen(params['hid_dim']*5)
        y_widen = model_widen(x).numpy()
    assert np.allclose(model.get_network_architecture(), [3, 5, 5, 5, 2])
    assert np.allclose(model_widen.get_network_architecture(), [3, 25, 25, 25, 2])
    assert np.allclose(y, y_widen)
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test1()