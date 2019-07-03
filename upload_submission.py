import numerapi
import click
import os
#from final_utils import reconocer_tournamet
import numpy as np


def make_submission(name_account ,account, resumen_dict, tournament,PATH):
    #list of targets

    for target, info in resumen_dict.items():
        print(info)
        #get csv file
        
        submission, status, PATH = info['root'], info['status'], PATH
        
        name = f'{name_account} prediccion, file: {submission}'
        n_tournamet = tournament[target]
       
        
        if (status == 'waiting') or (status == 'fail'):
            print(f'uploading {name} \n')
            try:
                #trata de subir la prediccion
                submission_id = account.upload_predictions(file_path = f'{PATH}{submission}_submission_lgbm.csv', tournament=n_tournamet)
                resumen = account.submission_status(submission_id)
                status = 'ok'
                print(resumen)
            except:
                status = 'fail'
            resumen_dict[target]['status'] = status
            resumen_dict[target]['account'] = name_account
            resumen_dict[target]['resumen'] = resumen
        
        #escribe el status de el submission 
    
    # check submission status
    np.save(PATH, resumen_dict)

    return f'{name_account}, Done!'  


@click.command()
@click.argument('n_round')


def main(n_round):

    accounts = {'LAGERTHA' : {'public_id' : 'GSMVUEGEEGMCU2LRXB6G4NR4QOXDNHBF',
                'secret_key': 'QT6WBGGKWD4TADJK4IKMK3BU3XKC3FBC5AKCHLBYKKV5RF2BWWAIUU3YPTS57JUW',
                'PATH': f'../submission/round {n_round}/RF&LG/resumen_info.npy'} 
                }
    
    ##list of targets
    tournament = {'target_bernie':1, 'target_elizabeth':2, 
        'target_jordan':3, 'target_ken':4, 'target_charles':5, 'target_frank':6 ,'target_hillary': 7}

    for name, acnt in accounts.items():
        
        current_acnt =  numerapi.NumerAPI(acnt['public_id'], acnt['secret_key'])
        resumen_dict = np.load(acnt['PATH']).item()
        # upload predictions
        resumen = make_submission( name, current_acnt, resumen_dict, tournament,acnt['PATH'])
        print(resumen)
    
    return 'Done!'
        

if __name__ == '__main__':
    main()