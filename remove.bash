#!/bin/bash          
##Foreground & background colour commands

#tput setab [1-7] # Set the background colour using ANSI escape
#tput setaf [1-7] # Set the foreground colour using ANSI escape
#Colours are as follows:

#Num  Colour    #define         R G B

#0    black     COLOR_BLACK     0,0,0
#1    red       COLOR_RED       1,0,0
#2    green     COLOR_GREEN     0,1,0
#3    yellow    COLOR_YELLOW    1,1,0
#4    blue      COLOR_BLUE      0,0,1
#5    magenta   COLOR_MAGENTA   1,0,1
#6    cyan      COLOR_CYAN      0,1,1
#7    white     COLOR_WHITE     1,1,1
#There are also non-ANSI versions of the colour setting functions (setb instead of setab, and setf instead of setaf) which use different numbers, not given here.

#Text mode commands

#tput bold    # Select bold mode
#tput dim     # Select dim (half-bright) mode
#tput smul    # Enable underline mode
#tput rmul    # Disable underline mode
#tput rev     # Turn on reverse video mode
#tput smso    # Enter standout (bold) mode
#tput rmso    # Exit standout mode
#Cursor movement commands

#tput cup Y X # Move cursor to screen postion X,Y (top left is 0,0)
#tput cuf N   # Move N characters forward (right)
#tput cub N   # Move N characters back (left)
#tput cuu N   # Move N lines up
#tput ll      # Move to last line, first column (if no cup)
#tput sc      # Save the cursor position
#tput rc      # Restore the cursor position
#tput lines   # Output the number of lines of the terminal
#tput cols    # Output the number of columns of the terminal
#Clear and insert commands

#tput ech N   # Erase N characters
#tput clear   # Clear screen and move the cursor to 0,0
#tput el 1    # Clear to beginning of line
#tput el      # Clear to end of line
#tput ed      # Clear to end of screen
#tput ich N   # Insert N characters (moves rest of line forward!)
#tput il N    # Insert N lines
#Other commands

#tput sgr0    # Reset text format to the terminal's default
#tput bel     # Play a bell

red=`tput setaf 1`
green=`tput setaf 2`
yellow=`tput setaf 3`
blue=`tput setaf 4`
bold=`tput bold`
reset=`tput sgr0`

echo "${yellow}${bold}Simple bash script to remove redundant files."

echo "The files found are as follows:"

echo "${blue} ${bold}.out files:${reset}${green}"
find . -name "*.out" -type f

echo "${blue} ${bold}.vtk files:${reset}${green}"
find . -name "*.vtk" -type f

echo "${blue} ${bold}.m files:${reset}${green}"
find . -name "*.m" -type f

echo "${blue} ${bold}.txt files:${reset}${green}"
find . -name "*.txt" -type f

echo "${blue} ${bold}fuse files:${reset}${green}"
find . -name ".fuse_*" -type f

read -p "${bold}${red}Would you like to delete those files?[Y/N](default N): ${yellow}" answer
if [[ $answer != y && $answer != Y ]]; then
  echo "${reset}${green}Aborting. Files will not be deleted.${reset}"
else
  echo "${red}${bold}Deleting .out files.${reset}"
	find . -name "*.out" -type f -delete
  sleep 0.25s
  echo "${red}${bold}Deleting .vtk files.${reset}"
	find . -name "*.vtk" -type f -delete
	sleep 0.25s  
	echo "${red}${bold}Deleting .m files.${reset}"
	find . -name "*.m" -type f -delete
 	sleep 0.25s
	echo "${red}${bold}Deleting .txt files.${reset}"
	find . -name "*.txt" -type f -delete
 	sleep 0.25s
	echo "${red}${bold}Deleting .fuse_* files.${reset}"
	find . -name ".fuse_*" -type f -delete
 	sleep 0.25s
  echo "${yellow}${bold}Files successfully deleted! Goodbye!${reset}"
fi 
