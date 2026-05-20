# Tec Dropouts

## Configuración

Para hacer más sencillo colaborar compartiendo por Git, usamos nbstripout, que filtra los
archivos cuando entran a git para quitar los outputs de las celdas, y así simplifica los
diffs, porque de otra forma se vuelve super gacho estar haciendo commits entre todos a las
notebooks. https://github.com/kynan/nbstripout

Para configurar nbstripout, recomiendo usar:
```bash
pip install nbstripout

nbstripout --install --python python3 --attributes .gitattributes
```

También recomiento aprovechar para correr:

```bash
pip install -r notebooks/requirements.txt
```

Si ya se tiene Jupyter instalado, ya está todo listo.
