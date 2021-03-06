DEST='/vol/bitbucket/mb4617/MRI_Crohns_Extended'
for a in 'A' 'I' 'a'
do
  for ext in '.nii.gz' '.nii'
  do
    for i in {1..100}
    do
      BASE=${DEST}/${a^}/${a}${i}
      BASE_D=${DEST}/${a^}/${a^}${i}

      mv ${BASE}\ T2\ Ax${ext}              ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\ T2\ Axial${ext}           ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\ t2\ ax${ext}              ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\ T2\ axial${ext}           ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\ T2\ AX${ext}              ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\ T2\ axial\ HASTE${ext}    ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\ T2\ Axial\ HASTE${ext}    ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\ T2\ HASTE\ axial${ext}    ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\ Ax\ T2${ext}              ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\ axial\ HASTE${ext}        ${BASE_D}\ Axial\ T2${ext}
      mv ${BASE}\.\ Axial\ T2${ext}         ${BASE_D}\ Axial\ T2${ext}

      mv ${BASE}\ T2\ Cor${ext}             ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\ T2\ Coronal${ext}         ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\ t2\ cor${ext}             ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\ T2\ coronal${ext}         ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\ T2\ COR${ext}             ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\ T2\ Coronal\ HASTE${ext}  ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\ T2\ coronal\ HASTE${ext}  ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\ T2\ HASTE\ coronal${ext}  ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\ Cor\ T2${ext}             ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\ coronal\ HASTE${ext}      ${BASE_D}\ Coronal\ T2${ext}
      mv ${BASE}\.\ Coronal\ T2${ext}       ${BASE_D}\ Coronal\ T2${ext}

      mv ${BASE}\ Postcon\ Ax${ext}            ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Post\ contrast\ Axial${ext}  ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ axial\ post\ contrast${ext}  ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Axial\ Post\ contrast${ext}  ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Axial\ Post\ Contrast${ext}  ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Postcon\ Axial${ext}         ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ POSTCON\ ax${ext}            ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Postcon\ AX${ext}            ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ POSTCON\ Ax${ext}            ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ axial\ postcon${ext}         ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Axial\ postcon${ext}         ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Axial\ postcontast${ext}     ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Axial\ postcontrast${ext}    ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\.\ Axial\ Postcontrast${ext}  ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Ax\ Postcon${ext}            ${BASE_D}\ Axial\ Postcon${ext}
      mv ${BASE}\ Postcontrast${ext}           ${BASE_D}\ Axial\ Postcon${ext}


      mv ${BASE}\ Postcon\ Ax\ UPPER${ext}           ${BASE_D}\ Axial\ Postcon\ Upper${ext}
      mv ${BASE}\ Postcon\ Ax\ LOWER${ext}           ${BASE_D}\ Axial\ Postcon\ Lower${ext}
      mv ${BASE}\ Postcontrast\ upper${ext}          ${BASE_D}\ Axial\ Postcon\ Upper${ext}
      mv ${BASE}\ Postcontrast\ lower${ext}          ${BASE_D}\ Axial\ Postcon\ Lower${ext}
      mv ${BASE}\ Post\ contrast\ Axial\ upper${ext} ${BASE_D}\ Axial\ Postcon\ Upper${ext}
      mv ${BASE}\ Post\ contrast\ Axial\ lower${ext} ${BASE_D}\ Axial\ Postcon\ Lower${ext}
      mv ${BASE}\ Axial\ Postcontrast\ upper${ext}   ${BASE_D}\ Axial\ Postcon\ Upper${ext}
      mv ${BASE}\ Axial\ Postcontrast\ lower${ext}   ${BASE_D}\ Axial\ Postcon\ Lower${ext}
      mv ${BASE}\.\ Axial\ Postcontrast\ upper${ext} ${BASE_D}\ Axial\ Postcon\ Upper${ext}
      mv ${BASE}\.\ Axial\ Postcontrast\ lower${ext} ${BASE_D}\ Axial\ Postcon\ Lower${ext}
    done
  done
done
