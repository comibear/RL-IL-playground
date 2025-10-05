#!/usr/bin/env bash
set -euo pipefail

# RL 폴더 내부 구조를 2단계 깊이로 출력
TREE=$(tree RL -L 2 --dirsfirst --charset utf-8)

# 새 블록 구성
NEWBLOCK=$(cat <<EOF
<!-- BEGIN TREE -->
\`\`\`text
$TREE
\`\`\`
<!-- END TREE -->
EOF
)

# README.md 내 트리 섹션 교체
awk -v repl="$NEWBLOCK" '
  /<!-- BEGIN TREE -->/ { print repl; inblock=1; next }
  /<!-- END TREE -->/   { inblock=0; next }
  inblock               { next }
  { print }
' README.md > README.tmp

mv README.tmp README.md
