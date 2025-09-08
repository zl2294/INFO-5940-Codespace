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

# echo 'export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"' >> ~/.bashrc
# echo 'export OPENAI_BASE_URL="https://api.ai.it.cornell.edu/"' >> ~/.bashrc
# echo 'export TZ="America/New_York"' >> ~/.bashrc
# source ~/.bashrc
