;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((nil
  (eval setenv "PYTHONPATH"
        (cdr
         (project-current))))
 (python-mode
  (rsync-remote-base-dir . "VAUXITE2:/esat/vauxite/xma/sing_images/")
  (eval add-to-list 'python-shell-extra-pythonpaths
        (cdr
         (project-current)))
  (eval venv-workon "dp")
  (eval message "Now in dp virtualenv")))
