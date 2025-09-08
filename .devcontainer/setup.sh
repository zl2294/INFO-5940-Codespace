sudo apt-get update 
sudo apt-get install -y \
    curl \
    git \
    unzip \
    vim \
    wget \
    gcc \

pip install -r requirements.txt

curl "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o "awscliv2.zip" \
    && unzip -o awscliv2.zip \
    && sudo ./aws/install --update \
    && echo 'complete -C '/usr/local/bin/aws_completer' aws' >> ~/.bashrc \
    && rm -rf awscliv2.zip ./aws


