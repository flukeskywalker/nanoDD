from d3pm import D3PM


def d3pm_text8_D6():
    model_args = dict(
        vocab_size=27,
        n_embed=768,
        n_heads=768 // 64,
        n_blocks=6,
        n_cond=128,
        dropout=0.0,
        T=1000,
        lambda_ce=0.01,
    )
    return D3PM, model_args
