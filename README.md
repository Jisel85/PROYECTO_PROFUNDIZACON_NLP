# PROYECTO_PROFUNDIZACON_NLP
NLP-LLM project.

# Configuración máquina google cloud

vm instance
    GPU NVIDIA L4 X 4
    100GB disco
    instancia spot en asia

# instalar vscode online
wget https://github.com/coder/code-server/releases/download/v4.16.1/code-server_4.16.1_amd64.deb
sudo dpkg -i code-server_4.16.1_amd64.deb 


# configuración vscode
nano ~/.config/code-server/config.yaml
# puerto 8123
# copiar password: 1eb5f4d15f38783a8595bb0e

# iniciar vscode online
sudo systemctl enable --now code-server@$USER
sudo systemctl start code-server@$USER
# verificar status
sudo systemctl status code-server@$USER



# instalar tunel https
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok

# agregar llaves para el tunel
ngrok config add-authtoken 2DhEI55Us80Ovqfu2J28ZMxhPmb_2roBsGTCYgvNh1rFVHj2b

# iniciar tunel
ngrok http --domain=correctly-dashing-kite.ngrok-free.app 8123

# ir a
https://correctly-dashing-kite.ngrok-free.app/?folder=/home/ladyotavo/nlp
pip install langchain[all]



### validar si el driver funciona
nvidia-smi

### POR SI SE DAÑA EL DRIVER
curl -O https://storage.googleapis.com/nvidia-drivers-us-public/GRID/vGPU16.1/NVIDIA-Linux-x86_64-535.104.05-grid.run
sudo bash NVIDIA-Linux-x86_64-535.104.05-grid.run --uninstall
sudo apt-get install linux-headers-`uname -r`
sudo bash NVIDIA-Linux-x86_64-535.104.05-grid.run

# utilizar sudo 
# se puede directamente desde el ssh de gcp
# desde vscode:
su root
# password: 12341234

# install docker
sudo apt update
sudo apt install docker-compose

# iniciar mongo
docker-compose up -d

# Extension VScode para ver la base de datos mongo: 
mongodb.mongodb-vscode

## Conexión al servidor de MongoDB (en este caso, está corriendo localmente)
client = MongoClient("mongodb://localhost:27017/")


## Entorno virtual

export PIPENV_VENV_IN_PROJECT=1

# install python3.10 https://computingforgeeks.com/how-to-install-python-on-debian-linux/
# error with bitsandbytes https://github.com/TimDettmers/bitsandbytes/issues/620

pip install pipenv 
pipenv install # or python3 -m pipenv install