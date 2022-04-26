import tensorflow as tf


def stable_log1pex(x):
    return -tf.minimum(x, 0) + tf.math.log(1+tf.math.exp(-tf.abs(x)))


def fake_negative_weighted_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x, (-1,))
    z = tf.cast(z, tf.float32)
    p_no_grad = tf.sigmoid(tf.stop_gradient(x))
    pos_loss = (1+p_no_grad)*stable_log1pex(x)
    neg_loss = -(1-p_no_grad)*(1+p_no_grad)*(-x-stable_log1pex(x))
    loss = tf.reduce_mean(pos_loss*z + neg_loss*(1-z))
    return {"loss": loss}

def unbiased_fnw_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x, (-1,))
    z = tf.cast(z, tf.float32)
    p_no_grad = tf.sigmoid(tf.stop_gradient(x)) 

    one = tf.constant([1.])
    oloss = 'DEFUSE'
    p_no_grad = tf.sigmoid(tf.stop_gradient(x))
    wi = p_no_grad # FN_prob
    loss1 = stable_log1pex(x) # DP
    loss2 = x + stable_log1pex(x)  # RN
    loss3 = stable_log1pex(x) # FN
    loss1_weight = one
    loss2_weight = one + p_no_grad
    loss3_weight = p_no_grad
    loss1 = loss1_weight * loss1
    loss2 = loss2_weight * loss2 * (one - wi)
    loss3 = loss3_weight * loss3 * wi
    # outw_loss = tf.reduce_mean(outw_label * loss1 + (one - outw_label) * (loss2)) # + loss3)) #+ loss4)
    loss = tf.reduce_mean(z * loss1 + (one - z) * (loss2 + loss3)) #+ loss4)
    # outw_loss = tf.reduce_mean(outw_label * loss1 + (one - outw_label) * (loss2 + loss3) + loss4)
    # loss = outw_loss
    return {
        "loss": loss
    }


def cross_entropy_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x, (-1,))
    z = tf.cast(z, tf.float32)
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=x))
    return {"loss": loss}


def exp_delay_loss(targets, outputs, params=None):
    z = tf.reshape(tf.cast(targets["label"][:, 0], tf.float32), (-1, 1))
    x = outputs["logits"]
    lamb = tf.math.softplus(outputs["log_lamb"])
    log_lamb = tf.math.log(lamb)
    d = tf.reshape(tf.cast(targets["label"][:, 1], tf.float32), (-1, 1))
    e = d
    p = tf.nn.sigmoid(x)
    pos_loss = -(-stable_log1pex(x) + log_lamb - lamb*d)
    neg_loss = -tf.math.log(1 - p + p*tf.math.exp(-lamb*e))
    return {"loss": tf.reduce_mean(pos_loss*z + neg_loss*(1-z))}


def delay_tn_dp_loss(targets, outputs, params=None):
    tn = tf.cast(outputs["tn_logits"], tf.float32)
    dp = tf.cast(outputs["dp_logits"], tf.float32)
    z = tf.cast(targets["label"], tf.float32)
    tn_label = tf.reshape(z[:, 0], (-1, 1))
    dp_label = tf.reshape(z[:, 1], (-1, 1))
    pos_label = tf.reshape(z[:, 2], (-1, 1))
    tn_mask = (1-pos_label)+dp_label
    tn_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tn_label, logits=tn)*tn_mask)\
        / tf.reduce_sum(tn_mask)
    dp_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=dp_label, logits=dp))
    loss = tn_loss + dp_loss
    return {
        "loss": loss,
        "tn_loss": tn_loss,
        "dp_loss": dp_loss
    }


def delay_tn_importance_weight_loss(targets, outputs, ep=None, dic=None, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]
    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    prob = tf.stop_gradient(tf.math.sigmoid(x))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    pos_loss = stable_log1pex(x)
    neg_loss = x + stable_log1pex(x)

    pos_weight = 1+dp_prob
    neg_weight = (1+dp_prob)*dist_prob
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss,
            "pos_weight": pos_weight,
            "neg_weight": neg_weight}


def unbiased_defuse_loss(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"] # p(y) p(d > e)
    # dp_logits2 = outputs["dp_logits2"]  # p(d > e)
    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    dp_label = targets["delay_label"]
    dp_label = tf.reshape(tf.cast(dp_label, tf.float32), (-1, 1))
    inw_label = targets["inw_label"]
    inw_label = tf.reshape(tf.cast(inw_label, tf.float32), (-1, 1))
    prob = tf.stop_gradient(tf.math.sigmoid(x))
    dist_prob = tf.stop_gradient(tf.math.sigmoid(tn_logits))
    dp_prob = tf.stop_gradient(tf.math.sigmoid(dp_logits))
    # dp_prob2 = tf.stop_gradient(tf.math.sigmoid(dp_logits2))

    pos_threshold = 0.7
    # zi = dp_label
    zi1 = 1 - dist_prob
    zi2 = (dp_prob) / (dp_prob + 1 - prob)
    zi = zi1

    one = tf.constant([1.])

    loss1_weight = tf.stop_gradient(one + dp_prob) # dp_prob: p(d > e, y=1 | x)
    loss2_weight = tf.stop_gradient(dp_prob)
    loss3_weight = tf.stop_gradient(one + dp_prob)
    loss4_weight = tf.stop_gradient(one)
    loss1 = -tf.math.log(tf.sigmoid(x))# * (1 - dp_prob2)) # IP: dp_prob2: p(d > e | y=1, x)
    loss2 = -tf.math.log(tf.sigmoid(x))# * dp_prob2) # FN
    loss3 = x + stable_log1pex(x) # RN
    loss4 = -tf.math.log(tf.sigmoid(x))# * dp_prob2) # DP
    loss1 = loss1 * loss1_weight
    loss2 = zi * loss2 * loss2_weight
    loss3 = (1 - zi) * loss3 * loss3_weight
    loss4 = loss4 * loss4_weight

    loss = tf.reduce_mean(z * (loss1 + loss4) + (1 - z) * (loss2 + loss3))# + loss5)
    return {"loss": loss,
            "pos_weight": loss1_weight + loss4_weight,
            "neg_weight": loss2_weight + loss3_weight}

def delay_dp_loss(targets, outputs, params=None):                                                                                                                                                                    
    dp = tf.reshape(tf.cast(outputs['logits'], tf.float32), [-1, 1])                                                                                                                                     
    z = tf.cast(targets["label"], tf.float32)                                                                                                                                              
    dp_label = tf.reshape(z, [-1, 1])                                                                                                                                                         
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=dp_label, logits=dp))                                                                                                      
    return {"loss": loss}

def delay_defer_loss(targets, outputs, ep=None, dic=None, params=None):                                                                                                                                                                      
    # return cross_entropy_loss(targets, outputs, params)
    z = targets["label"]
    x = outputs['logits']
    x = tf.reshape(x, (-1,))
    dp_logits = outputs['dp_logits']
    dp_logits = tf.reshape(dp_logits, (-1, ))
    z = tf.cast(z, tf.float32)

    p = tf.nn.sigmoid(x)                                                                                                                                                                            
    dp_prob = tf.math.sigmoid(dp_logits)                                                                                                                                                            
                                                                                                                                                                                                    
    # pos_weight = p / (1 - 0.5 * dp_prob)
    pos_weight = 2 * p / p * (2 - dp_prob)
    neg_weight = (1 - p) / (1 - p + 0.5*dp_prob)
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)
    # pos_weight = 1.0
    # neg_weight = 1.0

    pos_loss = stable_log1pex(x)                                                                                                                                                              
    neg_loss = x + stable_log1pex(x)
                                                                                                                                                                                                    
    clf_loss = tf.reduce_mean(pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))                                                                                                                    
    loss = clf_loss                                                                                                                                                                                 
    return {'loss': loss,
            "pos_weight": pos_weight,
            "neg_weight": neg_weight}

def unbiased_defer_loss(targets, outputs, ep=None, dic=None, params=None):                                                                                                                                                                      
    z = targets["label"]
    x = outputs['logits']
    tn_logits = outputs['tn_logits']
    x = tf.reshape(x, (-1,))
    dp_logits = outputs['dp_logits']
    dp_logits = tf.reshape(dp_logits, (-1, ))
    z = tf.cast(z, tf.float32)

    p = tf.nn.sigmoid(x)                                                                                                                                                                            
    dp_prob = tf.math.sigmoid(dp_logits)                                                                                                                                                            
    dist_prob = tf.stop_gradient(tf.math.sigmoid(tn_logits))

    zi = 1 - dist_prob
    # wi = tf.cast(wi > sample, tf.float32)
    
    # dp+fn=2 ; rn = ip = 2

    one = tf.constant([1.])
    loss1_weight = tf.stop_gradient(one + dp_prob) # dp_prob: p(d > e, y=1 | x)
    loss2_weight = tf.stop_gradient(dp_prob)
    loss3_weight = tf.stop_gradient(one + dp_prob)
    loss4_weight = tf.stop_gradient(one)
    # loss1_weight = 2
    # loss2_weight = 1
    # loss3_weight = 2
    # loss4_weight = 1
    loss1 = -tf.math.log(tf.sigmoid(x))# IP: * (1 - dp_prob2)) # dp_prob2: p(d > e | y=1, x)
    loss2 = -tf.math.log(tf.sigmoid(x))# FN: * dp_prob2)
    loss3 = x + stable_log1pex(x) # RN
    loss4 = -tf.math.log(tf.sigmoid(x))# DP: * dp_prob2)
    loss1 = loss1 * loss1_weight
    loss2 = zi * loss2 * loss2_weight
    loss3 = (1 - zi) * loss3 * loss3_weight
    loss4 = loss4 * loss4_weight

    loss = tf.reduce_mean(z * (loss1 + loss4) + (1 - z) * (loss2 + loss3))# + loss5)
    return {"loss": loss,
            "pos_weight": loss1_weight + loss4_weight,
            "neg_weight": loss2_weight + loss3_weight}

    
def delay_tn_importance_weight_loss10(targets, outputs, params=None):
    z = targets["label"]
    x = outputs['logits']
    x = tf.reshape(x, (-1,))
    dp_logits = outputs['dp_logits']
    dp_logits = tf.reshape(dp_logits, (-1, ))
    z = tf.cast(z, tf.float32)

    prob = tf.stop_gradient(tf.math.sigmoid(x))
    dp_prob = tf.math.sigmoid(dp_logits)

    #pos_loss = stable_log1pex(x)
    #neg_loss = x + stable_log1pex(x)

    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    py0 = tf.stop_gradient(1-tf.math.sigmoid(x))
    py1 = 1-py0

    pos_weight = py1 + 2*py0 + dp_prob
    neg_weight = (py1 + 2*py0 + dp_prob)*py0/(2*py0+dp_prob)
    #pos_weight = 1+dp_prob+py0
    #neg_weight = (1+dp_prob+py0)*dist_prob/2
    #pos_weight = 1+dp_prob
    #neg_weight = (1+dp_prob)*dist_prob/2
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = -tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}


def inw_outw_cross_entropy_loss(targets, outputs, params=None):
    inw_logits = tf.cast(outputs["logits_inw"], tf.float32)
    outw_logits = tf.cast(outputs["logits_outw"], tf.float32)
    #fusion_logits = tf.cast(outputs["fusion"], tf.float32)
    z = tf.cast(targets["label"], tf.float32)
    inw_label = tf.reshape(z[:, 0], (-1, 1))
    outw_label = tf.reshape(z[:, 1], (-1, 1))
    cvr_label = tf.reshape(z[:, 2], (-1, 1))
    inw_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=inw_label, logits=inw_logits))
    outw_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=outw_label, logits=outw_logits))
    #fusion_loss = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(labels=cvr_label, logits=fusion_logits))
    loss = inw_loss + outw_loss# + fusion_loss
    return {
        "loss": loss,
        "inw_loss": inw_loss,
        "outw_loss": outw_loss,
        #"fusion_loss": fusion_loss
    }

def unbiased_bidefuse_loss(targets, outputs, params=None):
    inw_logits = tf.cast(outputs["logits_inw"], tf.float32)
    outw_logits = tf.cast(outputs["logits_outw"], tf.float32)
    # fusion_logits = tf.cast(outputs["fusion"], tf.float32)

    cvr_label = tf.cast(targets["label"], tf.float32)
    cvr_label = tf.reshape(tf.cast(cvr_label, tf.float32), (-1, 1))

    outw_label = targets["outw_label"]
    outw_label = tf.reshape(tf.cast(outw_label, tf.float32), (-1, 1))

    inw_mask = targets["inw_mask"]
    inw_mask = tf.reshape(tf.cast(inw_mask, tf.float32), (-1, 1))
    # inw_label = cvr_label - outw_label

    inw_pos = stable_log1pex(inw_logits)
    inw_neg = inw_logits + stable_log1pex(inw_logits)
    inw_loss = tf.reduce_sum((cvr_label * inw_pos + (1 - cvr_label) * inw_neg) * inw_mask) \
        / tf.reduce_sum(inw_mask)
    
    #TODO
    one = tf.constant([1.])
    oloss = 'DEFUSE'
    if oloss == 'DEFUSE':
        p_no_grad = tf.sigmoid(tf.stop_gradient(outw_logits))
        wi = p_no_grad
        loss1 = stable_log1pex(outw_logits)
        loss2 = outw_logits + stable_log1pex(outw_logits)
        loss3 = stable_log1pex(outw_logits)
        loss1_weight = one
        loss2_weight = one + p_no_grad
        loss3_weight = p_no_grad
        loss1 = loss1_weight * loss1
        loss2 = loss2_weight * loss2 * (one - wi)
        loss3 = loss3_weight * loss3 * wi
        loss4 = ((outw_label * loss1_weight) + (one - outw_label) * ((one - wi) * loss2_weight + wi * loss3_weight)) \
            * tf.math.log(one + tf.math.sigmoid(outw_logits))
        # outw_loss = tf.reduce_mean(outw_label * loss1 + (one - outw_label) * (loss2)) # + loss3)) #+ loss4)
        outw_loss = tf.reduce_mean(outw_label * loss1 + (one - outw_label) * (loss2 + loss3)) #+ loss4)
        # outw_loss = tf.reduce_mean(outw_label * loss1 + (one - outw_label) * (loss2 + loss3) + loss4)
    elif oloss == 'ce':
        pos_loss = stable_log1pex(outw_logits)
        neg_loss = outw_logits + stable_log1pex(outw_logits)
        outw_loss = tf.reduce_mean(outw_label * pos_loss + (one - outw_label) * neg_loss)
        # outw_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=outw_label, logits=outw_logits))

    # fusion_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=cvr_label, logits=fusion_logits))
    loss = inw_loss + outw_loss # + fusion_loss
    # loss = outw_loss
    return {
        "loss": loss,
        "inw_loss": inw_loss,
        "outw_loss": outw_loss
    }


def get_loss_fn(name):
    if name == "cross_entropy_loss":
        return cross_entropy_loss
    elif name == "fake_negative_weighted_loss":
        return fake_negative_weighted_loss
    elif name == "unbiased_fake_negative_weighted_loss":
        return unbiased_fnw_loss 
    elif name == "delayed_feedback_loss":
        return exp_delay_loss
    elif name == "tn_dp_pretraining_loss":
        return delay_tn_dp_loss
    elif name == "dp_loss":
        return delay_dp_loss
    elif name == "esdfm_loss":
        return delay_tn_importance_weight_loss
    elif name == "defer_loss":
        return delay_defer_loss 
    elif name == "unbiased_defer_loss":
        return unbiased_defer_loss
    elif name == "defuse_loss":
        return unbiased_defuse_loss
    elif name == "bidefuse_loss":
        return unbiased_bidefuse_loss
    elif name == "inw_outw_cross_entropy_loss":
        return inw_outw_cross_entropy_loss
    else:
        raise NotImplementedError("{} loss does not implemented".format(name))
