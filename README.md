# rllib-policies

Repo to build RLlib policies.


# Implementing new policies 

An policy contains one or more networks. Each network processes a set of specified observations (e.g., images, graphs, poses). 

To build a new policy: 

1. Inheret from [NetworkBase](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/base.py#L10) to define a network in PyTorch. 

2. Inheret from [RllibPolicy](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/base.py#L96) to define a polciy. 
   - Any custom networks must be initialized in [`init_nets`](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/base.py#L142).
   - There are predefined actor-critic policies [with](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/actor_critic.py#L61) and [without](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/actor_critic.py#L15) a recurrent network
 
See [NatureCNNRNNActorCritic](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies/blob/develop/src/rllib_policies/vision.py#L92) for an example. 

# Examples

To use the CNN-LSTM Actor-Critic policy defined [here](https://github.mit.edu/aiia-suas-disaster-response/rllib_policies/blob/develop/src/rllib_policies/vision.py#L92)

```python

from rllib_policies.vision import NatureCNNRNNActorCritic
ModelCatalog.register_custom_model("nature_cnn_rnn", NatureCNNRNNActorCritic)

model = {
    "custom_model": "nature_cnn_rnn",
    "max_seq_len": 200,
    # keywords to custom model
    "custom_model_config": {
        "rnn_type": "LSTM",
        "hidden_size": 512,
        # cnn specific args
        "fields": ["RGB_LEFT", "DEPTH"],  # keys in observation dictionary 
        "cnn_shape_chw": [4, 192, 256],
    },
}

```

# Acknowledgement
Research was sponsored by the United States Air Force Research Laboratory and the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
