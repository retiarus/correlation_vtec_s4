# cap394-01-datascience
Projeto para a disciplina de data science cap394 do INPE

### Resumo

Era uma vez um jovem chamado João, ele iria fazer uma viagem partindo de São Paulo, para Bueno Aires, em um moderno avião que já implementava técnicas de navegação baseado em sistemas de navegação via satélite, tal como o GPS. A viagem, ocorreu, durante um período de intensa atividade solar. Infelizmente, João não chegou ao local de destino, pois o avião precisou pousar em outra cidade relativamente distante. 

Pode-se perguntar o que ocorreu? Observe que em sistemas de navegação por satélite, existe uma dependência de que o sinal mantenha certas características, ao longo do seu caminho. Agora, este sinal irá percorrer diferentes camadas da atmosfera, e uma destas é a ionosfera, sendo que esta por sua vez consiste de um gás em processo constante de recombinação e ionização, fenômeno que depende da atividade solar. Por exemplo, pense em um lâmpada fluorescente, se não houver eletricidade a lampada ficara apagada, adicionado energia a lampada se acende, a eletricidade força o gás contido na lampada a sofrer ionização, se variarmos a tensão, podemos ter uma luz mais ou menos intensa. O mesmo acontece na ionosfera em decorrência das atividades do Sol. 

Por outro lado, a Terra apresenta um campo magnético natural, no caso da lampada tente aproximar imãs dela e note o que acontece, o qual leva ao surgimento de correntes elétricas na ionosfera, entretanto, o campo não é uniforme, assim, como a luz solar, não atinge toda a Terra de maneira uniforme, o que leva a certas peculiaridades nessas correntes, anomalias. Assim, tem-se que em certas situações aquele sinal pode ter que atravessar as anomalias, o que leva a algumas imprecisões, por exemplo, o avião sair do curso.

De um ponto de vista mais formal, a ionosfera é um sistema dinâmico em constante mudança, e os processos envolvidos ainda não são completamente conhecidos ou estabelecidos, de forma que abordagens usuais de modelagem, como a resolução de sofisticados sistemas de equações diferenciais, não permite contemplar a enorme riqueza de anomalias envolvidas. Então, precisa-se de uma abordagem diferente? Claro, porque, não utilizar os dados coletados para a construção de um modelo fenomenológico. Neste caso, explorando técnicas de aprendizagem de máquina, técnicas estatísticas, visualizações diferentes dos dados.

Felizmente, não é necessário tratar todas as anomalias possíveis, mas aquelas que apresentam danos de alguma forma a sociedade, como o exemplo acima, que neste caso, é cintilação da ionosfera, ou para os mais íntimos bolhas, que nada mais é do que uma região mais rarefeita em relação a sua vizinhança, pense por exemplo em uma depressão, com uma escala de tempo de poucas horas.

O dado inicial a ser tratado é o VTEC (ou conteúdo total de elétrons vertical). Este pode ser obtido, por solicitação, por exemplo no Embrace, mas existem outras redes que também tem o material disponível. Porém, é um dado pré processado, existem também, a opção de coletar dados não tratados, estes podem ser acessados e obtidos no ftp do IBGE. 

O dado vai consistir basicamente, de uma lista, indexada por dois elementos latitude e longitude, com um valor adicional que é o VTEC, coletados ao longo do tempo. Para este trabalho, optou-se por um intervalo de tempo, onde as atividades solares são mais intensas. Finalmente, o que se tem são dados espaços-temporais, isto é, um serial temporal de dados espaciais. 

O objetivo é estudar e analisar os dados de forma a extrair correlações, por exemplo, quais as melhores maneiras de identificar a anomalia? A intensidade da anomalia? Prever a anomalia, dado que ela ocorreu em alguma região.
