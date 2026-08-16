import copy


class FederatedServer:
    def __init__(self, global_model, device):
        self.global_model = global_model.to(device)
        self.device = device

    def aggregate(self, client_states, client_sizes):
        if not client_states:
            raise ValueError("client_states cannot be empty")

        if len(client_states) != len(client_sizes):
            raise ValueError(
                "client_states and client_sizes must have the same length"
            )

        total_samples = sum(client_sizes)

        if total_samples <= 0:
            raise ValueError("Total number of samples must be greater than 0")

        new_state = copy.deepcopy(client_states[0])

        for key in new_state.keys():
            new_state[key] = (
                client_states[0][key]
                * (client_sizes[0] / total_samples)
            )

            for i in range(1, len(client_states)):
                weight = client_sizes[i] / total_samples

                new_state[key] += (
                    client_states[i][key] * weight
                )

        self.global_model.load_state_dict(new_state)

        return self.global_model