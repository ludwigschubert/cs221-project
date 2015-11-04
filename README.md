# cs221project

# Infrastructure

You can access our development server at `cruncher.schubert.io`.

User: `cs221`
Password: `github`

## SSH config

To make login easier, I propose using this in your ssh config (`~/.ssh/config`):

```
Host cruncher
  Hostname cruncher.schubert.io
  User cs221
  Port 2222
  IdentityFile ~/.ssh/id_rsa.pub
```
You can use any `IdentityFile` you want. For example if you're using your GitHub SSH key, this line would read `IdentityFile ~/.ssh/github_rsa.pub`.

## Mobile Shell access

`mosh-server` is now configured, so once SSH access works for you you should also be able to:

```
mosh cruncher
```
