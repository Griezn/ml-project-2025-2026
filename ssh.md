# Using the Departmental Machines Remotely

This document is a guide for using the departmental machines remotely. You can access the machines with a secure connection through SSH. This connection can be used for executing commands on the remote machine, tunneling through a firewall, and transferring files.

## Installing the SSH Client

### Unix (Linux/macOS)

Linux, macOS, and other Unix systems usually come with OpenSSH pre-installed. You can run `ssh -V` in a terminal window to verify this. If OpenSSH is not installed, install the `openssh-client` package via your package manager.

### Windows

Modern Windows (10/11) comes with OpenSSH pre-installed. You can simply open **PowerShell** or **Command Prompt** and run `ssh`.

*Note: If you prefer a full Linux environment on Windows, we recommend [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux). Third-party tools like PuTTY or MobaXterm are no longer necessary for most users, though they can still be used if you prefer the legacy interface.*

## Authenticating with SSH Certificates (MFA)

The department no longer uses static SSH keys (public/private key pairs). Instead, we use temporary **SSH Certificates** that are valid for one day. To get a certificate, you must authenticate using your KU Leuven Multi-Factor Authentication (MFA).

### Prerequisites

* A valid **KU Leuven account** (r-number, u-number, or s-number).
* The **KU Leuven Authenticator** app set up on your smartphone.

### Step 1: Download the Certificate Tool

You need a small utility program to request a certificate from the department server. Download the tool for your OS from the [Departmental MFA Documentation](https://admin.kuleuven.be/icts/services/ssh-cert).

* **Windows:** Download `certagent.exe`.
* **Linux/macOS:** Download the `kmk` script.

### Step 2: Activate your Certificate

You must run this step **every day** before you start working, as certificates expire after a few hours.

**On Windows:**

1. Run the `certagent.exe` you downloaded (no installation needed).
2. The first time you run it, enter your KU Leuven username (e.g., `r0123456`).
3. You will be prompted to approve the login on your **KU Leuven Authenticator** app.
4. Once approved, the tool automatically loads the certificate into your Windows OpenSSH agent.

**On Linux / macOS:**

1. Open your terminal.
2. Run the `kmk` tool with your username:
```bash
# Replace with your actual path and username
KMK_USER=r0123456 /path/to/downloaded/kmk

```


3. Approve the login on your **KU Leuven Authenticator** app.

### Step 3: Verify

To verify that you have a valid certificate loaded, run:

```bash
ssh-add -L

```

If successful, you will see a long string starting with `ssh-rsa-cert...`.

**Troubleshooting:**

* **Expiration:** If your connection is rejected, your certificate has likely expired. Simply run the tool again to renew it.
* **Conflict with old keys:** If you have old static SSH keys (from `ssh-keygen`), they might conflict. It is recommended to remove them from your agent (`ssh-add -D`) or ensure your config prefers the certificate.

## Configuring SSH and Running Commands

### A Simple Connection

Once your certificate is active, you can connect to the login node by opening a terminal (or PowerShell) and running:

```bash
ssh r0123456@st.cs.kuleuven.be

```

If the session asks for a **password**, it did not find your SSH certificate. Logging in with a password will not work. Try running `ssh -vvv r0123456@st.cs.kuleuven.be` to debug.

### Configuration (Recommended)

You can avoid typing the full username and hostname every time by creating a **config file**.

* **Location:** `~/.ssh/config` (Linux/macOS) or `C:\Users\YourName\.ssh\config` (Windows).
* **Content:** Add the text below to the file.

```text
# 1. The Login Node (Bastion)
Host pcroom
    User r0123456     # REPLACE WITH YOUR R-NUMBER
    HostName st.cs.kuleuven.be
    PasswordAuthentication no
    IdentitiesOnly no

# 2. The Compute Nodes (via Proxy)
Host *.student.cs.kuleuven.be
    User r0123456     # REPLACE WITH YOUR R-NUMBER
    ProxyJump pcroom

```

### Accessing Compute Nodes

The server `st.cs.kuleuven.be` is a **login node**; it is not meant for running experiments. You must connect to a compute node (e.g., `aalst`, `brugge`, etc.).

With the configuration above, you can connect directly:

```bash
ssh aalst.student.cs.kuleuven.be

```

This automatically "jumps" through the `pcroom` login node using your credentials.

## List of Available Machines

The page [http://mysql.student.cs.kuleuven.be/](http://mysql.student.cs.kuleuven.be/) gives an overview of the available departmental machines and their current load.

This page is only accessible from within the KU Leuven network. You can reach it remotely by setting up an SSH tunnel:

```bash
ssh -L 10480:mysql.student.cs.kuleuven.be:443 pcroom

```

Now, open your browser and go to: [https://localhost:10480/](https://www.google.com/search?q=https://localhost:10480/). (Note: Make sure to include `https`).

## Remote Copying

To transfer files, use the `scp` command. This works in both Unix terminals and Windows PowerShell.

**Copying local file TO remote:**

```bash
scp -r ./my_code/ aalst.student.cs.kuleuven.be:/cw/lvs/NoCsBack/vakken/H0T25A/ml-project/students/r0123456/

```

**Copying remote file TO local:**

```bash
scp aalst.student.cs.kuleuven.be:/cw/lvs/NoCsBack/vakken/H0T25A/ml-project/students/r0123456/results.txt ./

```

*Tip: If you use VS Code, you can simply drag and drop files using the "Remote - SSH" extension.*

## Safely Running Experiments with "Screen"

Since network connections can drop, you should never run long training jobs directly in the terminal. Use `screen` (or `tmux`) to keep sessions alive in the background.

1. **Start a session:**
```bash
screen

```


2. **Detach (Leave running):** Press `Ctrl-a` then `d`.
You can now safely disconnect SSH; your code will keep running.
3. **Reattach (Resume):** Log back in and run:
```bash
screen -r

```



**Common Commands (`Ctrl-a` prefix):**

* `Ctrl-a` then `c`: Create new window.
* `Ctrl-a` then `n`: Next window.
* `Ctrl-a` then `d`: Detach.

## Useful Directories

**Home directory:** `/home/r0123456/`
Accessible from all machines. *Check quota with `quota`.*

**Local space:** `/tmp/`
Fast, local storage on the specific machine you are using. Cleaned regularly. Use this for heavy datasets or logs during training, then move results to your home dir.

**Course directory:** `/cw/lvs/NoCsBack/vakken/H0T25A/ml-project/students/r0123456`
Your submission folder. Accessible from all machines (but not the login node). **Limit: 50MB.**
