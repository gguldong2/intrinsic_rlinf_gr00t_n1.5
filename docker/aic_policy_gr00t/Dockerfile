# GR00T-based AIC policy container — verification build
#
# This image bundles the pixi/ROS2 side of the policy (aic_model + RunGR00T.py
# + gr00t_worker.py). The heavier dependencies — GR00T base model, fine-tuned
# checkpoint, and the RLinf python 3.11 venv — are bind-mounted from the host
# at runtime (see docker-compose-policy.yaml).
#
# For a self-contained submission image, vendor those inside this Dockerfile
# (see comments below).

FROM docker.io/library/ros:kilted-ros-core AS build

# --- pixi + system deps ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# --- aic source -----------------------------------------------------------
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model            /ws_aic/src/aic/aic_model
COPY aic_interfaces       /ws_aic/src/aic/aic_interfaces
COPY aic_utils            /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock  /ws_aic/src/aic/
COPY pixi_env_setup.sh    /ws_aic/src/aic/

# --- build pixi env (also installs aic_example_policies with RunGR00T) ----
SHELL ["/bin/bash", "-c"]
RUN cd /ws_aic/src/aic && pixi install --locked

# --- extra pip deps required by RunGR00T (host side) ---------------------
# zmq is needed by RunGR00T.py to talk to the worker. Pixi env's python
# does not ship pip by default, so bootstrap it via ensurepip first.
RUN /ws_aic/src/aic/.pixi/envs/default/bin/python -m ensurepip --upgrade && \
    /ws_aic/src/aic/.pixi/envs/default/bin/python -m pip install --no-input pyzmq

# --- ur5e stats injection helper ------------------------------------------
COPY docker/aic_policy_gr00t/inject_ur5e_stats.py /opt/inject_ur5e_stats.py

# --- entrypoint -----------------------------------------------------------
COPY docker/aic_policy_gr00t/entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

WORKDIR /ws_aic/src/aic
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_example_policies.ros.RunGR00T", "-p", "use_sim_time:=true"]

# ===========================================================================
# Self-contained submission build (commented — uncomment for final image):
#
#   COPY models/gr00t-n1.5-3b /opt/models/gr00t-n1.5-3b
#   COPY checkpoint/full_weights.pt /opt/checkpoints/full_weights.pt
#   COPY rlinf /opt/rlinf
#   ENV GROOT_MODEL_PATH=/opt/models/gr00t-n1.5-3b \
#       GROOT_CHECKPOINT_PATH=/opt/checkpoints/full_weights.pt \
#       RLINF_PATH=/opt/rlinf \
#       RLINF_PYTHON=/opt/rlinf/.venv/bin/python
#
# ===========================================================================
