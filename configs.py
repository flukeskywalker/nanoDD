from d3pm_absorbing import D3PMAbsorbing


def d3pm_text8():
    model_args = dict(
        vocab_size=27,
        n_embed=768,
        n_heads=768 // 64,
        n_blocks=12,
        n_cond=128,
        dropout=0.025,
        T=1000,
        lambda_ce=0.05,
    )
    training_args = dict(
        batch_size=256,
        learning_rate=1e-3,
        min_lr=1e-5,
        gradient_accumulation_steps=8,
        warmup_iters=2_500,
        max_iters=500_000,
        eval_iters=1000,
        weight_decay=0.1,
        training_seed=1,
    )
    return D3PMAbsorbing, model_args, training_args


def d3pm_text8_gpu():
    model, model_args, training_args = d3pm_text8()
    training_args["gradient_accumulation_steps"] = 1
    training_args["eval_iters"] = 125
    return model, model_args, training_args


def d3pm_openwebtext_32gpu():
    model_args = dict(
        vocab_size=50257,
        n_embed=768,
        n_heads=768 // 64,
        n_blocks=12,
        n_cond=128,
        dropout=0.0,
        T=1000,
        lambda_ce=0.05,
    )
    training_args = dict(
        dataset="openwebtext",
        batch_size=16,
        seq_len=1024,
        learning_rate=6e-4,
        min_lr=1e-5,
        gradient_accumulation_steps=2,
        warmup_iters=2_500,
        max_iters=500_000,
        eval_iters=50,
        weight_decay=0.1,
        training_seed=9,
    )
    return D3PMAbsorbing, model_args, training_args
