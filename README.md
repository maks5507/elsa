# TLDR

Authors: Maksim Eremeev, Wei-Lun Huang, Eric Spector, Jeffrey Tumminia

## Codestyle check

Before making a commmit / pull-request, please check the coding style by running the bash script in the `codestyle` directory. Make sure that your folder is included in `codestyle/pycodestyle_files.txt` list.

Your changes will not be approved if the script indicates any incongruities (this does not apply to 3rd-party code). 

Usage:

```bash
cd codestyle
sh check_code_style.sh
```

