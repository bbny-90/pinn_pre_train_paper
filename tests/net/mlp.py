import inspect

def test1():
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
        y = model(x)
    assert y.shape == (bsz, params['out_dim'])
    print(f"{inspect.stack()[0][3]} is passed")

if __name__ == "__main__":
    test1()