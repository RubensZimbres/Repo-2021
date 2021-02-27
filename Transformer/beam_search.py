import torch
def beam_search_decoder(posterior, k):
    """Beam Search Decoder
    Shape:
        posterior: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).
    """

    batch_size, seq_length, _ = posterior.shape
    log_post = posterior.log()
    log_prob, indices = log_post[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1)
        log_prob, index = log_prob.view(batch_size, -1).topk(k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
    return indices, log_prob

posterior = torch.softmax(torch.randn([20, 8, 200]), -1)
indices, log_prob = beam_search_decoder(posterior, 3)
