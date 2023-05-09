import pkgutil

# when running on kaggle
if pkgutil.find_loader("kaggle_secrets") is not None:
    from kaggle_secrets import UserSecretsClient
    wandb_key = UserSecretsClient().get_secret("wandb_key")
# when running locally
else:
    from get_key import wandb_key
