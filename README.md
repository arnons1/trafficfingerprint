trafficfingerprint
==================
This should run on Win32/64 and *nix variants.

Windows
-------
I have included links for Win64 installs below (they're more difficult to find, most are from the [Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/))

*nix
----
On Debian, you can run ```sudo apt-get install python-numpy``` for instance. You can also use ```pip``` or ```easy-install``` to compile the prerequisites on your machine.

Prerequisites:
==============
The order for the installation is rather important, try to follow it.

* [Python 3.3 with Tkinter and ttk](http://www.python.org/ftp/python/3.3.5/python-3.3.5.amd64.msi)
* [Numpy (For math calculations)](http://www.lfd.uci.edu/~gohlke/pythonlibs/tid72nv9/numpy-MKL-1.8.1.win-amd64-py3.3.exe)
* [Scipy (For scalar quantization)](http://www.lfd.uci.edu/~gohlke/pythonlibs/tid72nv9/scipy-0.14.0c1.win-amd64-py3.3.exe)
* dpkt (For PCAP file parsing - patched version for 3.3 is in the prerequisites folder - copy to C:\Python33\Lib\site-packages\dpkt)
* Matplotlib/Pyplot/Pylab (follow this order for installation):
  * [Six](http://www.lfd.uci.edu/~gohlke/pythonlibs/tid72nv9/six-1.6.1.win-amd64-py3.3.exe)
  * [PyParsing](http://www.lfd.uci.edu/~gohlke/pythonlibs/tid72nv9/pyparsing-2.0.2.win-amd64-py3.3.exe)
  * [pytz](http://www.lfd.uci.edu/~gohlke/pythonlibs/tid72nv9/pytz-2014.2.win-amd64-py3.3.exe)
  * [python-dateutil](http://www.lfd.uci.edu/~gohlke/pythonlibs/tid72nv9/python-dateutil-2.2.win-amd64-py3.3.exe)
  * [Matplotlib/Pyplot/Pylab](http://www.lfd.uci.edu/~gohlke/pythonlibs/tid72nv9/matplotlib-1.3.1.win-amd64-py3.3.exe)
* [scikit-learn (Metrics for hamming_loss and log_loss)] (http://www.lfd.uci.edu/~gohlke/pythonlibs/)

Recommended:
* Wolfram Mathematica (For displaying the graphs and fingerprints)

Installing
==========
After installing ALL OF THE PREREQUISITES, clone the repository by running
```bash
git clone https://github.com/arnons1/trafficfingerprint.git
```

Running
=======
In your shell, run
```bash
python tree.py
```
The UI should open.
![Basic UI](https://github.com/arnons1/trafficfingerprint/tree/master/screenshots/Screenshot 2014-05-31 15.37.37.png)

Pick the desired directories for training and hit "Start Training".

When training is done, more buttons should appear with various actions.
![UI with more buttons](https://github.com/arnons1/trafficfingerprint/tree/master/screenshots/Screenshot 2014-05-31 15.37.44.png)

Selecting a capture, you can pick "Compare capture w/all fingerprints" to receive a general analysis, showing the most probable match.
![UI with more buttons](https://github.com/arnons1/trafficfingerprint/tree/master/screenshots/Screenshot 2014-05-31 15.37.55.png)

If you wish to run a test against a specific fingerprint, choose that fingerprint from the second dropdown, and pick "Analyze just this fingerprint"

 