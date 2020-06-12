source ./venv/bin/activate

NR_OF_EXPERIMENTS_TO_RUN=11

printf "Please run this script from the root folder of this repository\n"
printf "e.g. ~/evoman-game-playing-competition$ scripts/reinforcement_learning.sh"

for ((i = 1; i <= NR_OF_EXPERIMENTS_TO_RUN; i++)); do
  printf "\n\nRunning experiment number %d\n\n" "$i"
  python "code/reinforcement_learning/main.py"
done
