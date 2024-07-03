# Use the custom Python 3.10 base image
FROM python-3.10-base

ENV MYRAG_PROFILES=local

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
# COPY . /app

# Set up pip to use a specific mirror
RUN mkdir -p ~/.pip && \
    echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.pip/pip.conf

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip3.10 install --upgrade pip
RUN /usr/local/bin/python3.10 -m pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8008

CMD ["/usr/local/bin/python3.10", "launcher.py"] 
