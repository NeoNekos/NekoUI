use std::rc::Rc;

use crate::runtime::subscription_store::SubscriptionKey;

#[derive(Debug)]
pub struct Subscription {
    pub(crate) _key: SubscriptionKey,
    pub(crate) _owner: Rc<()>,
}

impl Subscription {
    pub(crate) fn new(key: SubscriptionKey, owner: Rc<()>) -> Self {
        Self {
            _key: key,
            _owner: owner,
        }
    }
}
