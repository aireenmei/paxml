pip uninstall -y jax
pip install 'jax @ git+https://github.com/google/jax'
pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install orbax==0.0.19
pip uninstall -y tensorboard-plugin-wit
pip install tbp-nightly