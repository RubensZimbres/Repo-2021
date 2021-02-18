LOCAL

```
$ sudo snap install --classic code
$ source ./bin/activate
$ code
```
GCP

```
$ wget https://github.com/cdr/code-server/releases/download/2.1688-vsc1.39.2/code-server2.1688-vsc1.39.2-linux-x86_64.tar.gz
$ tar -xvf code-server2.1688-vsc1.39.2-linux-x86_64.tar.gz
$ cd code-server2.1688-vsc1.39.2-linux-x86_64
$ sudo PASSWORD=123123 ./code-server --port 8888

## Download latest VS Code
$ sudo apt install libgconf-2-4
$ sudo apt --fix-broken install
$ sudo dpkg -i code_XXXXXXXXXX_amd64.deb

$ sudo ./code-server --port 8888 --auth none --host 0.0.0.
```

CTRL+SHIFT+P

```
$ ext install donjayamanne.githistory

CTRL+SHIFT+P 
ext install ms-python.python / chg version to 2020.3
>Select Interpreter: 
/home/anaconda3/bin/python3.7
GCP /opt/anaconda3/bin/python
Python 3.7.3 64bits (base:conda)

FUCKING JOKE: Just make sure that the LAST line you run when using (Shift + Enter) is not indented. If the last line is indented, you will either get an indentation error or be shown your code again in output -> https://github.com/microsoft/vscode-python/issues/2837
```
