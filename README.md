# Char_RNN
## A character-level RNN

Inspired by the blog of A. Karpathy about Recurrent Neural Networks (*http://karpathy.github.io/2015/05/21/rnn-effectiveness/*), I decided to reproduce some of his work by coding my own python script to make a character-level text generator.

The way the text is generated by these models is quite simple, it's generating character one after another using conditionnal probabilities computed by looking at the *n* previous characters. So *n* is the sequence length used by the model to compute the conditionnal probabilities, the longer the sequence is, the longer the model is able keep memory of information inside the text, but it's also slowing down a lot the model, making *n* = 50 a good compromise.

**Char_RNN.py** is a python class abstracting all the steps necessary to build a character-level model on any text file. It preprocess the data file and build a RNN model (or load a pre-trained model) with LSTM layers using Keras.

It can then be trained using the **train_model** method (**Model_training.py** provides an example).

Once the model is trained, text can be generated using the method **generate_sequence** (**Sequence_generation.py** provides an example).

## Model trained

The model has been trained on three different .txt files :
- **shakespeare.txt** : All the work of Shakespeare, (*https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt*).
- **numpy.txt** : All the .py files from the Numpy repository concatenated (*https://github.com/numpy/numpy*).
- **all_hugo.txt** : All the poesie of Victor Hugo, (*https://www.poesies.net/hugo.html*).

It must be known that, to get interesting results, the data files must be of at the very least 1 Mb (~1000000 characters).
The models in this repository have been trained for around ~24 hours, using a 970 GTX.

## Results

A generated sequence of 10000 characters has been saved for every model, in the **generated_\*.txt** files

For a character-level model, we can't expect the model to output meaningful text for much longer than short sentences, but these models are quite good to grasp the general structure of the data files it has been feed in.

## Some generated text examples
### Shakespeare

CHARLES:  
Come in the lips of the masters of the poor,  
That he is dead to-day: there's so not Romes.  

YORK:  
Thus are thou noble, to warn on his thoughts!  
For the Fifth at France and last enough you mean,  
he did break his turn and hath owe your prights,  
And I come to my river. I will do my contright  
where, but a rose of my name and other dispatch.  

SIR TOBY BELCH:  
Boy, how, we would a reason we are worth one rag.  

KING JOHN:  
This is no sons so dead to slame you in it.  

CASSIUS:  
The body I stood and hear me now;  
Why made it men of night.  

### Hugo

Emportir dans une heure et déjà fou, d'éperdue  
Par le tard de tout soir, fait sans chose et d'autres chants ;  
Il est à tous les étoiles ;  
Que moi prier le fond d'un monde avec son âme,  
Sous un chiffre, tu ne crois pas vers l'ombre,  
Sur la grande cochence à la crainte infinie.  
La foule était la terre à lui qui ne répandaite !  
Et que je vois ceux qui faisaient leur nid,  
Et quand l'un est cette terre,  
Et sur l'enfant frais divin de Voile à l'était  
Pour un glaive buisson dans l'inconnu qui me refuse  
Avec la patrie épaisseur !  

### Numpy
```
def err_bytes():
    """
     Yind use the simmlin flugative for required tile

    Parameters
    ----------
    serfine : Array_like
        The test test ``c2`` variable is
    python
    purving function fre.

     ABF: Force Get_FileWe note but remove the `__charself__``.

    Parameters
    ----------
    x : [x, y_1, 2-1]

    Returns
    -------
    value : number.

    See Also
    --------
    if doctestCesturate is object is not None
        (isnumpy varname:

        """
        outmess('(items when `zP = np.explicit_scalar, buildmultiple_block)))]

    # test typeowrating to tests delenser
    if ndim - str 3:
        qrie = re.compile(l + _vars[n]))
        log, s, int = '', simple", False
        if '*' in forc.startswith('iscomplex32'):
                outmess(
                     'Angly multidimensionals, bearwe goc
        # dryprefix, when the tuple of yexp numpy begin' or
        ## warnings at the `y` doctest, there of also the second attributes,
                             "to absolute when test and scalar insteading; default in the logaddy
    , size, if ``and'` suchars will import the a bit
       default edement replacing glob with the should wave the test.

            
           "              in given_expect`:
                                ssup.repeat()
                            else:
                                        vyedens.params
                    elif ls[1]:
                                    f2py_funclist = 'module' + 'type'

        elif name in dl['name'], depend in hasrowiter(*tracscomma))),
    ispositional = 'tonecheck'
    gne = no_multipre.cake.set(self)

    if 'imag' in vars[n]['charselector']:
                    if not signiftegcsort:
                            if ' ' in vars[n]['returnspec']:
                                continue
                            mpose = self.sup(params == sup.vars[n])))),
                      'parameter': True'")
                if 3 in line():
                                    return self.right, make the tage

                        _assert_note = xrnd = get_func(x - 1)], mask_the=res,
                                      * by_shape + 11
                            dep = np.divide(11, 2).complex(line, default.default(rown))))),
                        mP
        funcicillables = np.finfo(np.float64).eps
        if i
        tmp = sys.version_match(n[gigrecb]))  # note append
    outmess['vars'] = []
    return args
```
