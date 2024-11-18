# small-audio-data-aug

## to run jupyter on the box

### on the box (ssh)

jupyter notebook --generate-config

jupyter notebook password

in ~/.jupyter/jupyter_notebook_config.py:

c.ServerApp.ip = '0.0.0.0'  # Allow connections from all IPs
c.ServerApp.port = 8888     # Specify a port for Jupyter
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = True

jupyter notebook

### on laptop

ssh -N -L localhost:8888:localhost:8888 user@<server_ip>

access http://localhost:8888 in browser
