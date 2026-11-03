"""
Gera os dois PDFs exigidos pelo trabalho a partir de uma das versões:
  Arquivo 1: código-fonte (com nome e matrícula)
  Arquivo 2: resultado da execução, com as iterações até a convergência

Uso (no terminal do PyCharm, na pasta do projeto):
    python gerar_pdfs.py newton_v1_analitico.py

Preencha NOME e MATRICULA no topo do arquivo da versão escolhida ANTES de rodar.
Biblioteca necessária: reportlab  (pip install reportlab)
"""
import os
import re
import subprocess
import sys

from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas


def le_identificacao(texto):
    nome = re.search(r'^NOME\s*=\s*"(.*)"', texto, re.M)
    mat = re.search(r'^MATRICULA\s*=\s*"(.*)"', texto, re.M)
    return (nome.group(1) if nome else "?"), (mat.group(1) if mat else "?")


def escreve_pdf(caminho, titulo, linhas, nome, matricula, paisagem=False,
                fonte=8.0, numerar=False):
    pagina = landscape(A4) if paisagem else A4
    larg, alt = pagina
    c = canvas.Canvas(caminho, pagesize=pagina)
    margem = 36
    entrelinha = fonte * 1.22
    topo = alt - 62
    base = 40
    por_pagina = int((topo - base) / entrelinha)
    total = max(1, (len(linhas) + por_pagina - 1) // por_pagina)

    for pg in range(total):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margem, alt - 30, "Análise Numérica 2026/2 - %s" % titulo)
        c.setFont("Helvetica", 9)
        c.drawString(margem, alt - 44, "Aluno(a): %s    Matrícula: %s" % (nome, matricula))
        c.line(margem, alt - 50, larg - margem, alt - 50)
        c.setFont("Courier", fonte)
        y = topo
        for k, linha in enumerate(linhas[pg * por_pagina:(pg + 1) * por_pagina]):
            n = pg * por_pagina + k + 1
            txt = ("%4d  " % n + linha) if numerar else linha
            c.drawString(margem, y, txt.replace("\t", "    "))
            y -= entrelinha
        c.setFont("Helvetica", 8)
        c.drawRightString(larg - margem, 22, "Página %d de %d" % (pg + 1, total))
        c.showPage()
    c.save()


def main():
    if len(sys.argv) != 2:
        print("Uso: python gerar_pdfs.py <arquivo_da_versao.py>")
        sys.exit(1)
    script = sys.argv[1]
    base = os.path.splitext(os.path.basename(script))[0]
    with open(script, encoding="utf-8") as f:
        texto = f.read()
    nome, matricula = le_identificacao(texto)
    if nome.startswith("SEU NOME") or matricula.startswith("SUA MATRICULA"):
        print("AVISO: preencha NOME e MATRICULA no topo de %s antes de entregar." % script)

    os.makedirs("saida", exist_ok=True)

    # Arquivo 1: código-fonte
    escreve_pdf(os.path.join("saida", "arquivo1_codigo_%s.pdf" % base),
                "Código-fonte (%s)" % os.path.basename(script),
                texto.splitlines(), nome, matricula, paisagem=False,
                fonte=7.6, numerar=True)

    # Arquivo 2: resultado da execução (captura a saída do programa)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    exec_ = subprocess.run([sys.executable, script], capture_output=True,
                           text=True, encoding="utf-8", env=env)
    if exec_.returncode != 0:
        print(exec_.stderr)
        sys.exit("Erro ao executar %s" % script)
    escreve_pdf(os.path.join("saida", "arquivo2_resultado_%s.pdf" % base),
                "Resultados (%s)" % os.path.basename(script),
                exec_.stdout.splitlines(), nome, matricula, paisagem=True,
                fonte=7.2, numerar=False)
    print("PDFs gerados na pasta 'saida'.")


if __name__ == "__main__":
    main()